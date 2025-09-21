def get_num_params(model, non_embedding=True, exclude_layers=None):
    """
    Return the number of parameters in the model.
    
    Args:
        model: PyTorch model
        non_embedding: Whether to exclude position embeddings
        exclude_layers: List of layer names to exclude from count
    """
    n_params = sum(p.numel() for p in model.parameters())
    
    if non_embedding:
        # Subtract position embeddings - check for your current structure
        if hasattr(model, 'pos_emb'):
            n_params -= model.pos_emb.weight.numel()
        elif hasattr(model, 'trf_blocks') and hasattr(model.trf_blocks, 'pos_emb'):
            n_params -= model.trf_blocks.pos_emb.weight.numel()
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wpe'):
            n_params -= model.transformer.wpe.weight.numel()
    
    if exclude_layers:
        for name, param in model.named_parameters():
            if any(layer in name for layer in exclude_layers):
                n_params -= param.numel()
    
    return n_params

def print_model_info(model, non_embedding=True):
    """Print detailed model information."""
    total_params = get_num_params(model, non_embedding=False)
    core_params = get_num_params(model, non_embedding=non_embedding)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Core parameters: {core_params:,}")
    print(f"Position embeddings: {total_params - core_params:,}")
    
    # Print parameter breakdown by layer type
    layer_counts = {}
    for name, param in model.named_parameters():
        # Extract layer type from parameter name
        if 'trf_blocks.' in name:
            # For your current structure: trf_blocks.tok_emb.weight, trf_blocks.pos_emb.weight, etc.
            layer_type = name.split('.')[1]  # Get the part after trf_blocks
        else:
            layer_type = name.split('.')[0]  # Get first part of name
        
        layer_counts[layer_type] = layer_counts.get(layer_type, 0) + param.numel()
    
    print("\nParameter breakdown:")
    for layer_type, count in layer_counts.items():
        print(f"  {layer_type}: {count:,}")