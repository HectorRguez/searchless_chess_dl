import matplotlib.pyplot as plt
import numpy as np
import re
import os

def parse_output_file(filename):
    """Parse the output file with step and loss information.
    
    Args:
        filename: Path to the output file
        
    Returns:
        Tuple of (steps, losses) arrays
    """
    steps = []
    losses = []
    
    # Regular expression to match the step and loss values
    pattern = r"step: (\d+) \| loss: ([-+]?\d*\.?\d+)"
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                match = re.match(pattern, line.strip())
                if match:
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    steps.append(step)
                    losses.append(loss)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return [], []
    
    return np.array(steps), np.array(losses)

def plot_training_progress(steps, losses, output_path=None):
    """Plot training progress.
    
    Args:
        steps: Array of training steps
        losses: Array of loss values
        output_path: If provided, save plot to this path
    """
    plt.figure(figsize=(12, 6))
    
    # Main plot: Loss over steps
    plt.plot(steps, losses, 'b-', alpha=0.7)
    
    # Add smoothed curve (moving average)
  
    # Add labels and title
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    

    # Add legend
    plt.legend()
    
    # Ensure tick labels are visible
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def main():
    # Set the filename
    filename = "output_baseline.txt"  # Change this to your actual file path
    
    # Parse the file
    steps, losses = parse_output_file(filename)
    
    if len(steps) == 0:
        print("No data found in the file.")
        return
    
    print(f"Found {len(steps)} data points.")
    
    # Plot the results
    plot_output = "training_progress.png"
    plot_training_progress(steps, losses, output_path=plot_output)

if __name__ == "__main__":
    main()