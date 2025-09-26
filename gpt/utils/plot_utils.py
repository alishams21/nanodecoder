import matplotlib.pyplot as plt


def plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, hellaswag_accuracies=None):
    """Plot training and validation losses, optionally including HellaSwag accuracy."""
    if hellaswag_accuracies is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses vs tokens seen
    ax1.plot(tokens_seen, train_losses, label="Training loss")
    ax1.plot(tokens_seen, val_losses, label="Validation loss")
    ax1.set_xlabel("Tokens seen")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Training and Validation Loss vs Tokens")
    ax1.grid(True, alpha=0.3)
    # Format y-axis to avoid scientific notation for reasonable loss values
    ax1.ticklabel_format(style='plain', axis='y')
    
    # Plot losses vs tokens seen
    ax2.plot(tokens_seen, train_losses, label="Training loss")
    ax2.plot(tokens_seen, val_losses, label="Validation loss")
    ax2.set_xlabel("Tokens seen")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.set_title("Training and Validation Loss vs Tokens")
    ax2.grid(True, alpha=0.3)
    # Format y-axis to avoid scientific notation for reasonable loss values
    ax2.ticklabel_format(style='plain', axis='y')
    
    # Plot HellaSwag accuracy if provided
    if hellaswag_accuracies is not None:
        # Ensure all lists have the same length
        min_length = min(len(epochs_tensor), len(tokens_seen), len(hellaswag_accuracies))
        epochs_tensor = epochs_tensor[:min_length]
        tokens_seen = tokens_seen[:min_length]
        hellaswag_accuracies = hellaswag_accuracies[:min_length]
        
        # Filter out zero values for cleaner plotting
        valid_indices = [i for i, acc in enumerate(hellaswag_accuracies) if acc is not None and acc > 0]
        if valid_indices:
            valid_epochs = [epochs_tensor[i] for i in valid_indices]
            valid_accuracies = [hellaswag_accuracies[i] for i in valid_indices]
            valid_tokens = [tokens_seen[i] for i in valid_indices]
            
            # Plot accuracy vs tokens seen
            ax3.plot(valid_tokens, valid_accuracies, 'g-o', label="HellaSwag Accuracy", markersize=4)
            ax3.set_xlabel("Tokens seen")
            ax3.set_ylabel("Accuracy")
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.set_title("HellaSwag Accuracy vs Tokens")
            ax3.grid(True, alpha=0.3)
            
            # Add human performance reference line
            ax3.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label="Human Performance (95%)")
            ax3.legend()
            
            # Format y-axis as percentage for better readability
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}%'))
    
    plt.tight_layout()