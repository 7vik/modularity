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
import plotly.express as px
import logging

from modularity import clusterability

# --- Configuration ---
MODEL_NAME = "google/gemma-1.1-2b-it" # Using 2b version for potentially faster loading/iteration
DATASET_NAME = "XythicK/Chemistry"
DATASET_CONFIG = None  # Chemistry dataset doesn't have a config
# Using a very small slice for quick testing/demonstration
DATASET_SLICE_TRAIN = "train[:180]"
DATASET_SLICE_TEST = "train[180:200]"  # Held-out set from Chemistry dataset
NUM_EPOCHS = 10
BATCH_SIZE = 2 # Keep small for memory constraints
MAX_SEQ_LENGTH = 256 # Reduce sequence length for memory
LEARNING_RATE = 1e-5 # Adjusted learning rate
LOSS_FILE = "train_losses.txt"
TARGET_LAYER_INDEX = -2 # Second to last layer
MLORA_RANK = 256 # Rank for MLoRA matrices
CLUSTERABILITY_WEIGHT = 10.0 # Weight for clusterability term in loss function


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Setup ---
def setup():
    """Load environment variables and set device."""
    load_dotenv()
    # hf_token = os.getenv("HF_TOKEN")
    hf_token = "xxxxx"
    if not hf_token:
        logging.warning("HF_TOKEN environment variable not found. Ensure you are logged in via huggingface-cli login or set the HF_TOKEN.")
    # Use HF_TOKEN if available (required for Gemma models)
    # Note: transformers automatically uses HF_TOKEN env var if present

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device

# --- MLoRA Implementation ---
class MLoRAAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, rank=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        
        logging.info(f"Initializing MLoRA with dims: input={input_dim}, output={output_dim}, rank={rank}")
        
        # MLoRA matrices
        self.A = nn.Linear(input_dim, rank, bias=False)
        self.B = nn.Linear(rank, rank, bias=False)
        self.C = nn.Linear(rank, output_dim, bias=False)
        
        # Initialize with small values
        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.normal_(self.B.weight, std=0.01)
        nn.init.normal_(self.C.weight, std=0.01)
        
        # Scale factor to prevent output from dominating the residual
        self.scaling = 0.1
        
        # ReLU activations for H1 and H2
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Log shape for debugging
        batch_size = x.size(0)
        seq_len = x.size(1) if x.dim() > 2 else 1
        
        # Reshape if needed to handle different input formats
        if x.dim() > 2:
            # If x is [batch_size, seq_len, hidden_dim]
            orig_shape = x.shape
            x_reshaped = x.view(-1, self.input_dim)
        else:
            # If x is already [batch_size, hidden_dim]
            x_reshaped = x
            
        # MLoRA forward pass: x -> A -> H1 -> B -> H2 -> C -> out
        h1 = self.activation(self.A(x_reshaped))
        h2 = self.activation(self.B(h1))
        out = self.C(h2)
        
        # Apply scaling
        out = out * self.scaling
        
        # Reshape back to original shape if needed
        if x.dim() > 2:
            out = out.view(orig_shape)
            
        return out
    
    def get_B_matrix(self):
        # Return B matrix for clusterability computation
        return self.B.weight

# --- Data Loading and Preprocessing ---

class ChemistryDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        logging.info("Tokenizing dataset...")
        skipped_count = 0
        for text in tqdm(texts, desc="Tokenizing"):
            # Basic check for valid text (ensure it's a string and not too short)
            if text and isinstance(text, str) and len(text.strip()) > 10:
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
                    # For Causal LM, labels are typically input_ids.
                    # Padding token ids in labels are replaced with -100 to be ignored in loss calculation.
                    labels = input_ids.clone()
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
                # Log or count texts that are None, not strings, or too short
                if not (text and isinstance(text, str)):
                    logging.debug(f"Skipping non-string or empty text entry: {type(text)}")
                elif len(text.strip()) <= 10:
                    logging.debug(f"Skipping short text entry: {text[:30]}...")
                skipped_count += 1

        if skipped_count > 0:
            logging.warning(f"Skipped {skipped_count} invalid or short text entries.")
        if not self.encodings:
            raise ValueError("No valid data processed. Check dataset content, filtering, and 'TEXT' field extraction.")


    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        # Return the dictionary directly, DataLoader will batch items
        return self.encodings[idx]

def load_and_prepare_data(tokenizer, dataset_slice, dataset_name_arg, dataset_config_arg, max_seq_length_arg, batch_size_arg):
    """Loads and preprocesses the dataset."""
    logging.info(f"Loading dataset: {dataset_name_arg}, Config: {dataset_config_arg}, Slice: {dataset_slice}...")
    try:
        # Use arguments for dataset name and config
        data_hf = load_dataset(dataset_name_arg, dataset_config_arg, split=dataset_slice, trust_remote_code=True)

        # Extract text data - ensure we get strings from the 'TEXT' field for the chemistry dataset
        # Also handle cases where 'TEXT' might be missing or not a string for robustness
        texts = []
        for item in data_hf:
            question_content = item.get('Question')
            answer_content = item.get("Answer_1")
            text_content = f"<start_of_turn>user\n{question_content}<end_of_turn>\n<start_of_turn>model\n{answer_content}<end_of_turn>"
            # text_content = question_content + answer_content
            if isinstance(text_content, str):
                texts.append(text_content)
            else:
                logging.debug(f"Found item without valid 'TEXT' field or non-string content: {item}")

        logging.info(f"Loaded {len(texts)} text documents from 'TEXT' field.")
        if not texts:
            raise ValueError("No text data found in the 'TEXT' field of the dataset slice.")
    except Exception as e:
        logging.error(f"Failed to load or process dataset: {e}")
        raise

    # Use argument for max_seq_length
    dataset = ChemistryDataset(texts, tokenizer, max_seq_length_arg)
    if len(dataset) == 0:
        raise ValueError("Dataset created, but contains no processable entries. Check data and tokenization.")

    # Shuffle training data, don't shuffle test data
    # This logic assumes specific slice names for train/test.
    # Consider a more robust way if slice names vary (e.g., pass a boolean is_train flag).
    is_train = "train[" in dataset_slice and ":180]" in dataset_slice # Based on your example slice
    # is_train = "train" in dataset_slice.lower() # A more general check

    dataloader = DataLoader(dataset, batch_size=batch_size_arg, shuffle=is_train) # Use argument for batch_size
    logging.info(f"Created DataLoader with {len(dataloader)} batches.")
    return dataloader



# --- Model Loading ---
def load_model_and_tokenizer():
    """Loads the model and tokenizer."""
    hf_token = "xxxxx
    logging.info(f"Loading tokenizer: {MODEL_NAME}")
    hf_token = os.getenv("HF_TOKEN")
    # trust_remote_code=True might be needed for some models
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token = hf_token)

    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Set pad_token to eos_token")

    logging.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        token = hf_token
        # Use torch_dtype=torch.float16 or bfloat16 for memory efficiency if GPU supports it
        # torch_dtype=torch.bfloat16,
        # device_map="auto" # Can help distribute large models across GPUs/CPU
    )
    # Ensure model's pad token id matches tokenizer's
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer



# --- Custom Forward Function with MLoRA ---
def model_forward_with_mlora(model, mlora_adapter, input_ids, attention_mask, labels=None):
    """
    Custom forward pass that applies MLoRA adapter to the target MLP's output.
    """
    # Prepare inputs
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    
    # If labels provided, add them to inputs
    if labels is not None:
        model_inputs["labels"] = labels
    
    # Get original outputs - we need to handle the hidden states
    model_inputs["output_hidden_states"] = True
    outputs = model(**model_inputs)
    
    # We need to modify this to return modified logits
    return outputs

# --- Custom Hook for MLoRA ---
def register_mlora_hook(model, mlora_adapter, target_layer_index):
    """
    Registers a forward hook on the target MLP layer to apply MLoRA.
    """
    # For Gemma, the path to the MLP output is model.model.layers[index].mlp
    target_layer = model.model.layers[target_layer_index].mlp
    
    # Get shape information for debugging
    logging.info(f"Registering hook on layer: {target_layer}")
    
    # We need to track whether this is the first forward pass to log dimension info
    first_pass = [True]
    
    def mlora_hook(module, input, output):
        """
        Hook that applies MLoRA to the MLP output.
        - input: tuple containing tensor of shape [batch_size, seq_len, hidden_dim]
        - output: tensor of shape [batch_size, seq_len, hidden_dim]
        """
        if first_pass[0]:
            logging.info(f"Hook input shape: {input[0].shape}")
            logging.info(f"Hook output shape: {output.shape}")
            first_pass[0] = False
        
        try:
            # Apply MLoRA adapter to the input tensor
            mlora_output = mlora_adapter(input[0])
            
            # Handle shape mismatch if needed
            if mlora_output.shape != output.shape:
                logging.warning(f"Shape mismatch: MLoRA output {mlora_output.shape}, MLP output {output.shape}")
                # Reshape to match output dimensions
                mlora_output = mlora_output.view_as(output)
            
            # Add residual connection
            return output + mlora_output
            
        except RuntimeError as e:
            # If there's an error, log it but return the original output to prevent training failure
            logging.error(f"Error in MLoRA hook: {e}")
            return output
    
    # Register the hook to run after the MLP forward pass
    hook_handle = target_layer.register_forward_hook(mlora_hook)
    return hook_handle

# --- Loss Function ---
def calculate_custom_loss(outputs, labels, mlora_adapter):
    """
    Calculates the custom loss with clusterability regularization.
    """
    # Calculate standard cross-entropy loss
    loss_fct = nn.CrossEntropyLoss() # Ignores index -100 by default
    logits = outputs.logits
    
    # Reshape for loss calculation
    ce_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    # Get B matrix from MLoRA adapter
    B_matrix = mlora_adapter.get_B_matrix()
    
    # Calculate clusterability term
    clusterability_term = clusterability(B_matrix)
    
    # Combined loss: CE - λ * Clusterability(B)
    total_loss = ce_loss - CLUSTERABILITY_WEIGHT * clusterability_term
    
    return total_loss, ce_loss, clusterability_term

# --- Evaluation Function ---
def evaluate_model(model, mlora_adapter, dataloader, device):
    """Evaluates the model on the provided dataloader."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():  # No gradient calculation during evaluation
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move batch tensors to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model_forward_with_mlora(
                model=model,
                mlora_adapter=mlora_adapter,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Calculate loss (CE only for evaluation)
            loss_fct = nn.CrossEntropyLoss()
            logits = outputs.logits
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss

# --- Training Logic ---
def train_model(model, mlora_adapter, train_dataloader, test_dataloader, optimizer, device, num_epochs):
    """Runs the custom fine-tuning loop with MLoRA."""
    model.train()  # Ensure model is in training mode
    all_losses = []
    all_ce_losses = []
    all_clusterability_values = []
    all_test_losses = []
    total_steps = len(train_dataloader) * num_epochs

    logging.info("Starting training...")
    logging.info(f"Total steps: {total_steps}")

    hook_handle = register_mlora_hook(model, mlora_adapter, TARGET_LAYER_INDEX)
    
    global_step = 0
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_ce_losses = []
        epoch_clusterability_values = []
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch in progress_bar:
            # Move batch tensors to the correct device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Zero gradients before forward pass
            optimizer.zero_grad()

            # Forward pass with MLoRA
            try:
                outputs = model_forward_with_mlora(
                    model=model,
                    mlora_adapter=mlora_adapter,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # Calculate custom loss with clusterability
                loss, ce_loss, clusterability_value = calculate_custom_loss(
                    outputs=outputs,
                    labels=labels,
                    mlora_adapter=mlora_adapter
                )

                # Backward pass (calculates gradients)
                loss.backward()

                # Optimizer step (updates parameters)
                optimizer.step()

                # Record losses
                batch_loss = loss.item()
                epoch_losses.append(batch_loss)
                all_losses.append(batch_loss)
                
                # Record component losses
                epoch_ce_losses.append(ce_loss.item())
                all_ce_losses.append(ce_loss.item())
                epoch_clusterability_values.append(clusterability_value.item())
                all_clusterability_values.append(clusterability_value.item())
                
                progress_bar.set_postfix({
                    'loss': batch_loss,
                    'ce_loss': ce_loss.item(),
                    'clust': clusterability_value.item()
                })
                global_step += 1

            except Exception as e:
                logging.error(f"Error during training step {global_step}: {e}")
                continue

        # Evaluate on test set after each epoch
        logging.info("Evaluating on test dataset...")
        test_loss = evaluate_model(model, mlora_adapter, test_dataloader, device)
        all_test_losses.append(test_loss)

        # Log epoch metrics
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        avg_epoch_ce_loss = sum(epoch_ce_losses) / len(epoch_ce_losses) if epoch_ce_losses else 0
        avg_epoch_clusterability = sum(epoch_clusterability_values) / len(epoch_clusterability_values) if epoch_clusterability_values else 0
        
        logging.info(f"Epoch {epoch+1} Metrics:")
        logging.info(f"  - Train Loss: {avg_epoch_loss:.4f}")
        logging.info(f"  - CE Loss: {avg_epoch_ce_loss:.4f}")
        logging.info(f"  - Clusterability: {avg_epoch_clusterability:.4f}")
        logging.info(f"  - Test Loss: {test_loss:.4f}")

    # Remove hook after training
    hook_handle.remove()
    
    return {
        'train_losses': all_losses,
        'ce_losses': all_ce_losses,
        'clusterability_values': all_clusterability_values,
        'test_losses': all_test_losses
    }

# --- Main Execution ---
def main():
    device = setup()
    model, tokenizer = load_model_and_tokenizer()
    model.to(device)

    # --- Identify Target Layer's MLP Parameters ---
    mlora_adapter = None
    try:
        # Get the target layer
        target_mlp_layer = model.model.layers[TARGET_LAYER_INDEX].mlp
        
        # Create a sample batch for tracing
        sample_text = "This is a sample text for dimension tracing."
        sample_encoding = tokenizer(
            sample_text,
            return_tensors="pt", 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        )
        sample_input_ids = sample_encoding["input_ids"].to(device)
        sample_attention_mask = sample_encoding["attention_mask"].to(device)
        
        # Run a forward pass to inspect model dimensions
        with torch.no_grad():
            # Get hidden states to inspect dimensions
            outputs = model(
                input_ids=sample_input_ids, 
                attention_mask=sample_attention_mask,
                output_hidden_states=True
            )
            
            # For Gemma models, hidden states usually have shape [batch_size, seq_len, hidden_size]
            # Get estimated dimensions from model config
            hidden_size = model.config.hidden_size
            intermediate_size = getattr(model.config, 'intermediate_size', hidden_size * 4)
            
            logging.info(f"Model hidden_size: {hidden_size}")
            logging.info(f"Model intermediate_size: {intermediate_size}")
            
            # Verify with actual tensor shapes if possible
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                # Get the hidden state before the target MLP layer
                target_idx = len(model.model.layers) + TARGET_LAYER_INDEX
                if 0 <= target_idx < len(outputs.hidden_states):
                    target_hidden = outputs.hidden_states[target_idx]
                    logging.info(f"Target hidden state shape: {target_hidden.shape}")
                    input_dim = target_hidden.shape[-1]
                else:
                    input_dim = hidden_size
            else:
                input_dim = hidden_size
            
            output_dim = hidden_size  # Usually MLP outputs match the model's hidden size
            
        # Create MLoRA adapter with appropriate dimensions
        # If MLORA_RANK is too large, it might cause shape issues
        # Ensure rank is not larger than the smaller of input_dim and output_dim
        safe_rank = min(MLORA_RANK, min(input_dim, output_dim) // 2)
        if safe_rank != MLORA_RANK:
            logging.warning(f"Reducing MLoRA rank from {MLORA_RANK} to {safe_rank} to ensure dimensional compatibility")
        
        logging.info(f"Creating MLoRA adapter with input_dim={input_dim}, output_dim={output_dim}, rank={safe_rank}")
        mlora_adapter = MLoRAAdapter(
            input_dim=input_dim,
            output_dim=output_dim,
            rank=safe_rank
        )
        mlora_adapter.to(device)
        
    except Exception as e:
        logging.error(f"Error initializing MLoRA adapter: {e}")
        logging.error("Attempting fallback initialization...")
        
        # Fallback to simpler initialization with model's hidden size
        try:
            hidden_size = model.config.hidden_size
            safe_rank = min(8, hidden_size // 4)  # Use a much smaller rank as fallback
            
            logging.info(f"Fallback: Creating MLoRA adapter with input_dim={hidden_size}, output_dim={hidden_size}, rank={safe_rank}")
            mlora_adapter = MLoRAAdapter(
                input_dim=hidden_size,
                output_dim=hidden_size,
                rank=safe_rank
            )
            mlora_adapter.to(device)
        except Exception as inner_e:
            logging.error(f"Fallback initialization also failed: {inner_e}")
            return
    
    if mlora_adapter is None:
        logging.error("Failed to initialize MLoRA adapter. Exiting.")
        return

    # --- Freeze All Model Parameters ---
    logging.info("Freezing all model parameters...")
    for param in model.parameters():
        param.requires_grad = False
    
    # --- Setup Optimizer for MLoRA only ---
    optimizer = optim.AdamW(mlora_adapter.parameters(), lr=LEARNING_RATE)
    logging.info(f"Optimizer configured with {sum(p.numel() for p in mlora_adapter.parameters() if p.requires_grad)} trainable parameters in MLoRA adapter.")

    # --- Load Data ---
    try:
        # train_dataloader = load_and_prepare_data(tokenizer, DATASET_SLICE_TRAIN)
        # test_dataloader = load_and_prepare_data(tokenizer, DATASET_SLICE_TEST)
        train_dataloader = load_and_prepare_data(
            tokenizer, DATASET_SLICE_TRAIN, DATASET_NAME, DATASET_CONFIG, MAX_SEQ_LENGTH, BATCH_SIZE
        )
        test_dataloader = load_and_prepare_data(
            tokenizer, DATASET_SLICE_TEST, DATASET_NAME, DATASET_CONFIG, MAX_SEQ_LENGTH, BATCH_SIZE
        )

        logging.info(f"Loaded train dataloader with {len(train_dataloader)} batches")
        logging.info(f"Loaded test dataloader with {len(test_dataloader)} batches")
    except ValueError as e:
        logging.error(f"Failed to prepare data: {e}")
        return

    # --- Train ---
    try:
        training_results = train_model(
            model=model,
            mlora_adapter=mlora_adapter,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            device=device,
            num_epochs=NUM_EPOCHS
        )
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return

    # --- Save Results ---
    logging.info("Saving training results...")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save training metrics
    metrics = {
        'train_losses': training_results['train_losses'],
        'ce_losses': training_results['ce_losses'],
        'clusterability_values': training_results['clusterability_values'],
        'test_losses': training_results['test_losses']
    }
    
    try:
        with open(f"{results_dir}/training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info("Training metrics saved successfully.")
    except IOError as e:
        logging.error(f"Error saving metrics: {e}")

    # --- Save MLoRA Adapter ---
    try:
        torch.save(mlora_adapter.state_dict(), f"{results_dir}/mlora_adapter.pt")
        logging.info("MLoRA adapter saved successfully.")
    except IOError as e:
        logging.error(f"Error saving MLoRA adapter: {e}")

    # --- Visualize Results ---
    os.makedirs("figures", exist_ok=True)
    
    # Plot training and test loss
    fig_loss = px.line(
        x=list(range(len(training_results['train_losses']))), 
        y=training_results['train_losses'],
        title="Training Loss"
    )
    fig_loss.write_image(file="figures/train_losses.pdf", format="pdf")
    
    # Plot CE loss and clusterability values
    fig_components = px.line(
        x=list(range(len(training_results['ce_losses']))),
        y=[training_results['ce_losses'], training_results['clusterability_values']],
        title="Loss Components",
        labels={'value': 'Value', 'variable': 'Component'},
        color_discrete_sequence=['blue', 'red']
    )
    fig_components.write_image(file="figures/loss_components.pdf", format="pdf")
    
    # Plot test loss per epoch
    fig_test = px.line(
        x=list(range(1, NUM_EPOCHS + 1)),
        y=training_results['test_losses'],
        title="Test Loss per Epoch",
        labels={'x': 'Epoch', 'y': 'Test Loss'}
    )
    fig_test.write_image(file="figures/test_losses.pdf", format="pdf")

    logging.info("Fine-tuning finished.")

if __name__ == "__main__":
    main() 