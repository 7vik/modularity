import os
import yaml
import torch
import logging
import numpy as np
import pickle as pkl
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib.colors import to_rgb, rgb_to_hsv
from transformer_lens.evals import make_wiki_data_loader


# Configuration for Pythia 70m
CONFIG = {
    "model_name": "EleutherAI/pythia-70m",
    "model": "jvelja/pythia-finetune-pythia-70m-clusters-4-NoBSGC-lr_5e-05-Modularity-WIKIFixCluster",
    "nmodel": "jvelja/pythia-finetune-pythia-70m-NoBSGC-lr_5e-05-NoModularity-WIKIFixCluster",
    "log_file": "interpretability/logs/pythia70m_model_analysis_on_ravel.log",
    "plot_path": "interpretability/module_analysis/plots/pythia70m",
    "data_path": "interpretability/module_analysis/data/pythia70m"
}


class ModelIntervention:
    """
    Class for neural network module intervention analysis on Pythia 70m.
    
    Performs two types of interventions:
    1. Switch off 1 module (keep 3 modules on)
    2. Switch off 3 modules (keep only 1 module on)
    """
    
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.device = None
        self.hook_ = None
        
        # Will be populated during analysis
        self.list1_filtered = None
        self.list2_filtered = None
        self.list3_filtered = None
        self.list4_filtered = None
    
    def setup_model(self):
        """Initialize model and tokenizer for Pythia 70m."""
        self.device = torch.device(self.args.device)
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            CONFIG[self.args.modeltype], 
            device_map=self.args.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
        
        # Configure logging
        logging.basicConfig(
            filename=CONFIG["log_file"],
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.info(f"Loaded model: {self.model.__class__.__name__}")
        
        return self.model, self.tokenizer, self.device
    
    def register_hook(self, index):
        """Register a hook to perform the intervention on a specific layer."""
        def hook_fn(module, input, output):
            # Create a copy of the output to modify
            mod_output = output.clone()
            
            if index == "baseline":
                return output
            
            # Apply zeroing based on index configuration
            if len(index) == 2:
                # Handle case where second index is None
                if index[1] is not None:
                    mod_output[:, :, index[0]:index[1]] = 0
                else:
                    mod_output[:, :, index[0]:] = 0
            elif len(index) == 4:
                # First range
                if index[1] is not None:
                    mod_output[:, :, index[0]:index[1]] = 0
                else:
                    mod_output[:, :, index[0]:] = 0
                
                # Second range
                if index[3] is not None:
                    mod_output[:, :, index[2]:index[3]] = 0
                else:
                    mod_output[:, :, index[2]:] = 0
                    
            return mod_output

        # Register hook for Pythia 70m
        self.hook_ = self.model.gpt_neox.layers[self.args.num_layer].mlp.dense_h_to_4h.register_forward_hook(hook_fn)
    
    def analyze_predictions(self, module, samples):
        """Analyze model predictions with the specified intervention."""
        predictions = []
        correct_samples = []
        
        for sample_idx in tqdm(range(len(samples)), desc=f"Analyzing {module}"):
            sample = samples[sample_idx].to(self.device)
            
            # Get model predictions
            logits = self.model(sample)[0]
                
            # Compare predicted token with ground truth
            is_correct = sample[:, -1].item() == logits[:, -2, :].argmax(dim=-1).item()
            predictions.append(1 if is_correct else 0)
            
            if is_correct:
                correct_samples.append(sample)

        # Save predictions and correct samples
        data_path = f"{CONFIG['data_path']}/{self.args.modeltype}"
        os.makedirs(data_path, exist_ok=True)
        
        prediction_file = f"{data_path}/prediction_{self.args.type_of_intervention}_layer{self.args.num_layer}_{module}.pkl"
        samples_file = f"{data_path}/samples_{self.args.type_of_intervention}_layer{self.args.num_layer}_{module}.pkl"
        
        with open(prediction_file, "wb") as f:
            pkl.dump(predictions, f)
        
        with open(samples_file, "wb") as f:
            pkl.dump(correct_samples, f)
            
        return predictions
    
    def run_final_analysis(self):
        """Analyze results across all modules and create visualizations."""
        final_dict = {}
        mean_accuracies = []
        
        # Create main accuracy plot
        plt.figure(figsize=(10, 5))
        
        # Process each module's results
        for module in ["mod1", "mod2", "mod3", "mod4"]:
            prediction_file = (
                f"{CONFIG['data_path']}/{self.args.modeltype}/"
                f"prediction_{self.args.type_of_intervention}_layer{self.args.num_layer}_{module}.pkl"
            )
            
            with open(prediction_file, "rb") as f:
                predictions = pkl.load(f)
            
            final_dict[module] = predictions
            mean_accuracy = np.mean(predictions)
            mean_accuracies.append(mean_accuracy)
            
            print(f"{module} mean accuracy: {mean_accuracy:.4f}")
        
        # Plot mean accuracies
        plt.plot(mean_accuracies, marker="s", color="orange", markersize=10)
        plt.title("Mean Accuracy by Module", size=16)
        plt.xlabel("Module Index", size=12)
        plt.ylabel("Accuracy", size=12)
        plt.xticks(range(4), ["Module 1", "Module 2", "Module 3", "Module 4"])
        plt.grid(True)
        
        # Save the accuracy plot
        plot_path = f"{CONFIG['plot_path']}/{self.args.modeltype}"
        os.makedirs(plot_path, exist_ok=True)
        
        accuracy_plot = f"{plot_path}/{self.args.type_of_intervention}_layer{self.args.num_layer}_accuracy.png"
        plt.savefig(accuracy_plot, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Process data for module dependency analysis
        self._process_module_dependencies(final_dict)
    
    def _process_module_dependencies(self, final_dict):
        """Process module dependencies and prepare for visualization."""
        # Invert results (1 = error, 0 = correct)
        # This shows when turning off a module causes an error
        module_errors = {}
        for mod in ["mod1", "mod2", "mod3", "mod4"]:
            module_errors[mod] = 1 - np.array(final_dict[mod])
        
        # Keep only samples that show dependency on at least one module
        filtered_lists = [
            (a, b, c, d) for a, b, c, d in zip(
                module_errors['mod1'], 
                module_errors['mod2'],
                module_errors['mod3'],
                module_errors['mod4']
            )
            if not (a == 0 and b == 0 and c == 0 and d == 0)
        ]
        
        # Skip if no meaningful results
        if not filtered_lists:
            print("No samples showed module dependencies")
            return
            
        # Unzip filtered tuples into separate lists for each module
        self.list1_filtered, self.list2_filtered, self.list3_filtered, self.list4_filtered = map(list, zip(*filtered_lists))
        
        # Create pie chart visualization
        self.create_dependency_pie_chart()
    
    def create_dependency_pie_chart(self):
        """Create a pie chart showing the distribution of module dependencies."""
        # Count dependencies for pie chart categories
        all_four = 0  # Samples that depend on all 4 modules
        all_three = 0  # Samples that depend on exactly 3 modules
        all_two = 0   # Samples that depend on exactly 2 modules
        all_one = 0   # Samples that depend on just 1 module

        # Count samples in each category
        for idx in range(len(self.list1_filtered)):
            # Sum of 1s across all modules for this sample
            dependency_count = (
                self.list1_filtered[idx] + 
                self.list2_filtered[idx] + 
                self.list3_filtered[idx] + 
                self.list4_filtered[idx]
            )
            
            if dependency_count == 4:
                all_four += 1
            elif dependency_count == 3:
                all_three += 1
            elif dependency_count == 2:
                all_two += 1
            elif dependency_count == 1:
                all_one += 1

        # Data for the pie chart
        labels = ['Depends on all 4', 'Depends on 3', 'Depends on 2', 'Depends on 1']
        sizes = [all_four, all_three, all_two, all_one]
        colors = ['#267326', '#5db85d', '#91d891', '#d4f7d4']  # Green gradient (dark to light)

        # Configure matplotlib for better styling
        mpl.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'legend.fontsize': 12,
        })

        # Create the pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 14}
        )

        # Adjust text colors for readability
        for i, autotext in enumerate(autotexts):
            autotext.set_color(self._get_contrasting_color(colors[i]))
            autotext.set_weight("bold")
            
        ax.axis('equal')  # Equal aspect ratio for circular pie chart
        plt.title("Module Dependency Distribution", pad=20)

        # Save the chart
        output_path = f"{CONFIG['plot_path']}/{self.args.modeltype}/pie_chart"
        os.makedirs(output_path, exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(
            f"{output_path}/{self.args.type_of_intervention}_layer{self.args.num_layer}_pie.png", 
            dpi=400, 
            bbox_inches='tight'
        )
        plt.close()
    
    def _get_contrasting_color(self, hex_color):
        """Return white for dark colors and black for light colors."""
        rgb = to_rgb(hex_color)
        brightness = rgb_to_hsv(rgb)[2]
        return "white" if brightness < 0.5 else "black"
    
    def prepare_dataset(self):
        """Prepare dataset by filtering for samples where baseline model is correct."""
        correct = 0
        total = 0
        samples = []
        
        # Create data loader
        data_loader = make_wiki_data_loader(
            self.tokenizer, 
            batch_size=self.args.batch_size
        )
        
        # Process batches to find samples where model is correct
        for idx, data_ in enumerate(tqdm(
            data_loader, 
            desc="Preparing dataset", 
            total=len(data_loader)
        )):
            data = data_['tokens'].to(self.device)
            
            # Get model predictions
            logits = self.model(data)[0]
                
            # Check if prediction is correct
            if data[:, -1].item() == logits[:, -2, :].argmax(dim=-1).item():
                correct += 1
                samples.append(data)
            total += 1
            
            if idx % 100 == 0:
                print(f"Baseline accuracy: {correct/total:.4f}")
        
        # Remove hook if it exists
        if hasattr(self, 'hook_') and self.hook_:
            self.hook_.remove()
        
        # Save filtered dataset
        data_path = f"{CONFIG['data_path']}/{self.args.modeltype}"
        os.makedirs(data_path, exist_ok=True)
        
        with open(f"{data_path}/cropped_dataset_last_token.pkl", "wb") as f:
            pkl.dump(samples, f)
        
        return samples
    
    def run_intervention(self, index, module, operation="analysis"):
        """Run intervention with specified parameters."""
        # Setup model and config if not already done
        if self.model is None:
            self.setup_model()
            
        # Register hook for intervention
        self.register_hook(index)
        
        # Try to load existing samples or prepare new ones
        try:
            with open(f"{CONFIG['data_path']}/{self.args.modeltype}/cropped_dataset_last_token.pkl", "rb") as f:
                samples = pkl.load(f)   
        except FileNotFoundError:
            print("Dataset not found, preparing new dataset...")
            samples = self.prepare_dataset()
        
        # Run the requested operation
        if operation == "analysis":
            self.analyze_predictions(module, samples)
        elif operation == "final_analysis":
            self.run_final_analysis()
            
        # Clean up hook
        if hasattr(self, 'hook_') and self.hook_:
            self.hook_.remove()


def create_visualization_collage(folder_paths):
    """Create collages of charts from multiple folders."""
    for folder_path in tqdm(folder_paths, desc="Creating collages"):
        # List all image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()  # Sort files for consistent order
        
        # Skip if no images found
        if not image_files:
            print(f"No images found in {folder_path}")
            continue
            
        # Set up the figure
        num_images = len(image_files)
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 4, 4))
        
        # Handle single image case
        if num_images == 1:
            axes = [axes]

        # Plot each image in the row
        for ax, img_file in zip(axes, image_files):
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')  # Turn off axes

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{folder_path}/collage.png", dpi=300, bbox_inches='tight')
        plt.close()


def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Pythia 70m module intervention analysis")
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda, cpu)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loading')
    parser.add_argument('--num_layer', type=int, default=6, help='Layer number to intervene on')
    parser.add_argument('--type_of_intervention', type=str, required=True, 
                        choices=['type1', 'type2'], help='Intervention type (type1 or type2)')
    parser.add_argument('--modeltype', type=str, required=True, 
                        choices=['model', 'nmodel'], help='Model type (model or nmodel)')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Define module indices based on intervention type
    if args.type_of_intervention == "type1":
        # Switch off one module at a time (1024//4 = 256)
        indices = {
            "mod1": [0, 256],        # Switch off first quarter
            "mod2": [256, 256*2],    # Switch off second quarter
            "mod3": [256*2, 256*3],  # Switch off third quarter
            "mod4": [256*3, None]    # Switch off fourth quarter
        }
    elif args.type_of_intervention == "type2":
        # Keep only one module on
        indices = {
            "mod1": [256, 256*3, 256*3, None],  # Switch on just 1st module
            "mod2": [0, 256, 256*2, None],      # Switch on just 2nd module
            "mod3": [0, 256*2, 256*3, None],    # Switch on just 3rd module
            "mod4": [0, 256*3, None, None]      # Switch on just 4th module
        }
    
    # Create intervention object
    intervention = ModelIntervention(args)
    
    # Run intervention for each module
    for i, (module, index) in enumerate(indices.items()):
        print(f"Running intervention on {module} (layer {args.num_layer})")
        
        # For the last module, also run final analysis
        operation = "final_analysis" if i == len(indices) - 1 else "analysis"
        intervention.run_intervention(index, module, operation)
    
    # Create visualization collages
    folder_paths = [
        f"{CONFIG['plot_path']}/model/pie_chart/type1",
        f"{CONFIG['plot_path']}/model/pie_chart/type2",
        f"{CONFIG['plot_path']}/nmodel/pie_chart/type1",
        f"{CONFIG['plot_path']}/nmodel/pie_chart/type2",
    ]
    create_visualization_collage(folder_paths)
    
    print("Analysis complete!")


if __name__ == '__main__':
    main()