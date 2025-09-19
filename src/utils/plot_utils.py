import matplotlib.pyplot as plt


def plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses):
    """Plot training and validation losses."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses vs epochs
    ax1.plot(epochs_tensor, train_losses, label="Training loss")
    ax1.plot(epochs_tensor, val_losses, label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Training and Validation Loss")
    
    # Plot losses vs tokens seen
    ax2.plot(tokens_seen, train_losses, label="Training loss")
    ax2.plot(tokens_seen, val_losses, label="Validation loss")
    ax2.set_xlabel("Tokens seen")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.set_title("Training and Validation Loss vs Tokens")
    
    plt.tight_layout()