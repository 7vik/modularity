import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import load_dataset
from tqdm.auto import tqdm
from dotenv import load_dotenv
import torch.nn as nn
import torch.optim as optim
import logging

# --- Configuration ---
MODEL_NAME = "google/gemma-1.1-2b-it" # Using 2b version for potentially faster loading/iteration
DATASET_NAME = "wikipedia"
DATASET_CONFIG = "20220301.en"
# Using a very small slice for quick testing/demonstration
DATASET_SLICE = "train[:200]"
NUM_EPOCHS = 10
BATCH_SIZE = 2 # Keep small for memory constraints
MAX_SEQ_LENGTH = 256 # Reduce sequence length for memory
LEARNING_RATE = 1e-4 # Adjusted learning rate
LOSS_FILE = "train_losses.txt"
TARGET_LAYER_INDEX = -2 # Second to last layer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Setup ---
def setup():
    """Load environment variables and set device."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logging.warning("HF_TOKEN environment variable not found. Ensure you are logged in via huggingface-cli login or set the HF_TOKEN.")
    # Use HF_TOKEN if available (required for Gemma models)
    # Note: transformers automatically uses HF_TOKEN env var if present

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device

# --- Data Loading and Preprocessing ---
class WikiDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        logging.info("Tokenizing dataset...")
        skipped_count = 0
        for text in tqdm(texts, desc="Tokenizing"):
            if text and isinstance(text, str) and len(text.strip()) > 10: # Basic check for valid text
                try:
                    # Tokenize, pad, and truncate
                    tokenized = tokenizer(
                        text,
                        truncation=True,
                        max_length=self.max_length,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    # Remove the batch dimension added by the tokenizer
                    input_ids = tokenized["input_ids"].squeeze(0)
                    attention_mask = tokenized["attention_mask"].squeeze(0)

                    # Create labels by cloning input_ids
                    labels = input_ids.clone()
                    # In Causal LM, labels are typically input_ids shifted right.
                    # We replace padding token ids in labels with -100 so they are ignored in loss calculation.
                    labels[labels == self.tokenizer.pad_token_id] = -100

                    self.encodings.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    })
                except Exception as e:
                    logging.warning(f"Skipping text due to tokenization error: {e}. Text snippet: {text[:100]}...")
                    skipped_count += 1
            else:
                 skipped_count += 1

        if skipped_count > 0:
            logging.warning(f"Skipped {skipped_count} invalid or short text entries.")
        if not self.encodings:
             raise ValueError("No valid data processed. Check dataset and filtering.")


    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        # Return the dictionary directly, DataLoader will batch items
        return self.encodings[idx]

def load_and_prepare_data(tokenizer):
    """Loads and preprocesses the dataset."""
    logging.info("Loading dataset...")
    try:
        wiki_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SLICE, trust_remote_code=True)
        # Extract text data - ensure we get strings
        texts = [item['text'] for item in wiki_dataset if isinstance(item.get('text'), str)]
        logging.info(f"Loaded {len(texts)} text documents.")
        if not texts:
             raise ValueError("No text data found in the dataset slice.")
    except Exception as e:
        logging.error(f"Failed to load or process dataset: {e}")
        raise

    dataset = WikiDataset(texts, tokenizer, MAX_SEQ_LENGTH)
    if len(dataset) == 0:
        raise ValueError("Dataset created, but contains no processable entries.")

    # Ensure shuffle=True for training data is standard practice
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    logging.info(f"Created DataLoader with {len(dataloader)} batches.")
    return dataloader

# --- Model Loading ---
def load_model_and_tokenizer():
    """Loads the model and tokenizer."""
    logging.info(f"Loading tokenizer: {MODEL_NAME}")
    # trust_remote_code=True might be needed for some models
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Set pad_token to eos_token")

    logging.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        # Use torch_dtype=torch.float16 or bfloat16 for memory efficiency if GPU supports it
        # torch_dtype=torch.bfloat16,
        # device_map="auto" # Can help distribute large models across GPUs/CPU
    )
    # Ensure model's pad token id matches tokenizer's
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

# --- Loss Function ---
def calculate_causal_lm_loss(outputs, labels):
    """
    Calculates the cross-entropy loss for Causal LM.
    Designed to be easily swappable if needed.
    """
    loss_fct = nn.CrossEntropyLoss() # Ignores index -100 by default
    logits = outputs.logits

    # Logits shape: [batch_size, seq_length, vocab_size]
    # Labels shape: [batch_size, seq_length]

    # Reshape logits to [batch_size * seq_length, vocab_size]
    # Reshape labels to [batch_size * seq_length]
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss

# --- Training Logic ---
def train_model(model, dataloader, loss_fn, optimizer, device, num_epochs):
    """Runs the custom fine-tuning loop."""
    model.train() # Ensure model is in training mode
    all_losses = []
    total_steps = len(dataloader) * num_epochs

    logging.info("Starting training...")
    logging.info(f"Total steps: {total_steps}")

    global_step = 0
    for epoch in range(num_epochs):
        epoch_losses = []
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch in progress_bar:
            # Move batch tensors to the correct device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Zero gradients before forward pass
            optimizer.zero_grad()

            # Forward pass
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels # Pass labels for HF models to potentially compute loss internally (though we compute manually)
                )

                # Calculate custom loss
                loss = loss_fn(outputs, labels)

                # Backward pass (calculates gradients)
                loss.backward()

                # Gradient Clipping (optional but recommended)
                # torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)

                # Optimizer step (updates parameters)
                optimizer.step()

                # Record loss
                batch_loss = loss.item()
                epoch_losses.append(batch_loss)
                all_losses.append(batch_loss)
                progress_bar.set_postfix({'loss': batch_loss})
                global_step += 1

            except Exception as e:
                 logging.error(f"Error during training step {global_step}: {e}")
                 # Optionally skip batch or raise error depending on severity
                 # raise e # Re-raise if critical
                 continue # Skip batch


        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        logging.info(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

    return all_losses

# --- Main Execution ---
def main():
    device = setup()
    model, tokenizer = load_model_and_tokenizer()
    model.to(device)

    # --- Identify Target Layer's MLP Parameters ---
    try:
        # Gemma structure: model.model.layers[index].mlp
        # MLP typically has gate_proj, up_proj, down_proj
        target_mlp_layer = model.model.layers[TARGET_LAYER_INDEX].mlp
        target_params_dict = dict(target_mlp_layer.named_parameters())

        if not target_params_dict:
             raise ValueError(f"No parameters found for MLP layer at index {TARGET_LAYER_INDEX}.")

        logging.info(f"Identified target MLP layer at index {TARGET_LAYER_INDEX}. Parameters to train:")
        for name, param in target_params_dict.items():
            logging.info(f"  - {name}: {param.shape}, Requires Grad: {param.requires_grad}")

    except (AttributeError, IndexError, ValueError) as e:
        logging.error(f"Error accessing target MLP layer: {e}")
        logging.error("Model structure might differ. Inspect the model architecture.")
        # print(model) # Uncomment to print model structure for debugging
        return # Exit if layer identification fails

    # --- Freeze Non-Target Layers ---
    logging.info("Freezing non-target layers...")
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        # Check if the parameter name matches one in our target MLP dictionary
        is_target_mlp_param = False
        # Need to check full path, e.g., model.layers.X.mlp.gate_proj.weight
        layer_prefix = f"model.layers.{len(model.model.layers) + TARGET_LAYER_INDEX}.mlp."
        if name.startswith(layer_prefix) and name.split(layer_prefix)[1] in target_params_dict:
             is_target_mlp_param = True

        if not is_target_mlp_param:
            param.requires_grad = False
        else:
            param.requires_grad = True # Explicitly ensure target params require grad
            trainable_params += param.numel()
            logging.info(f"Parameter {name} marked as trainable.")

    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters (target MLP): {trainable_params} ({100 * trainable_params / total_params:.4f}%)")

    if trainable_params == 0:
        logging.error("Error: No parameters were marked as trainable. Check layer identification and freezing logic.")
        return

    # --- Setup Optimizer ---
    # Filter parameters that require gradients
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    if not params_to_optimize:
         logging.error("Optimizer received no parameters to optimize. Check requires_grad flags.")
         return

    optimizer = optim.AdamW(params_to_optimize, lr=LEARNING_RATE)
    logging.info(f"Optimizer configured with {len(params_to_optimize)} trainable parameter tensors.")


    # --- Load Data ---
    try:
        dataloader = load_and_prepare_data(tokenizer)
    except ValueError as e:
        logging.error(f"Failed to prepare data: {e}")
        return

    # --- Train ---
    try:
        train_losses = train_model(
            model=model,
            dataloader=dataloader,
            loss_fn=calculate_causal_lm_loss, # Pass the modular loss function
            optimizer=optimizer,
            device=device,
            num_epochs=NUM_EPOCHS
        )
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return # Exit if training fails critically


    # --- Save Losses ---
    logging.info(f"Saving training losses to {LOSS_FILE}")
    try:
        with open(LOSS_FILE, 'w') as f:
            for loss_value in train_losses:
                f.write(f"{loss_value}
")
        logging.info("Losses saved successfully.")
        # Example: Save as JSON (alternative)
        # with open("train_losses.json", 'w') as f:
        #     json.dump({"losses": train_losses}, f, indent=4)
    except IOError as e:
        logging.error(f"Error saving losses to file {LOSS_FILE}: {e}")

    logging.info("Fine-tuning finished.")

if __name__ == "__main__":
    main() 