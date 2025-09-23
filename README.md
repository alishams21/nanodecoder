# Nanodecoder 

Nanodecoder is meant as a foundation for experimenting, and extending Transformer architectures. Modern large language models (LLMs) predominantly adopt a decoder-only transformer architecture, omitting separate encoder modules and cross-attention mechanisms typical of encoder-decoder designs. These decoder-only models rely solely on causal self-attention within stacked transformer blocks, enabling efficient and scalable autoregressive text generation. Such a structure is ideal for tasks that require prediction of the next token in a sequence, making it the backbone of systems like GPT-4 and Llama-2. 


## GPT Variants

Nanodecoder currently implements two different types of LLM architectures:

### Dense-GPT
- **Description**: All parameters are engaged for every token (e.g., GPT-2, GPT-3, GPT-4).  
- **This Repo Note**: Here the `dense-gpt` implementation is designed to be **beginner-friendly**, making it ideal for learning and understanding the core GPT architecture.  

### MoE-GPT
- **Description**: Follows the GPT style, but replaces certain dense feedforward layers with **Mixture-of-Experts (MoE)** layers (e.g., GPT-OSS, DeepSeek-V3, Switch Transformer).  
- **Repo Note**: The `moe-gpt` folder provides **production-ready code** with comprehensive feature support.  
- **Features table**:  

   | Feature | CPU | GPU | Mixed + TF32 | Multi-GPU + Compile
   |---------|-----|-----|-------|--------------|
   | CPU training | ✅ | ✅ | ✅ | ✅ | 
   | GPU training | ❌ | ✅ | ✅ | ✅ |
   | Mixed precision | ❌ | ❌ | ✅ | ✅ |
   | Multi-GPU DDP | ❌ | ❌ | ❌ | ✅ |
   | Model compilation | ❌ | ❌ | ❌ | ✅ |
   | Wandb logging | ❌ | ❌ | ❌ | ✅ |


- **If you are using GPU you need to consider following hardware requirements:**

   ```
   Working GPUs: Ampere architecture (2020+)
   # - RTX 30xx series (RTX 3090, 3080, 3070, etc.)
   # - A100, A40, A30
   # - RTX 4090, 4080, 4070

   Not supported:
   # - RTX 20xx series
   # - GTX series
   # - Older GPUs
   ```

## MOE-GPT Architecture

Following image shows the architecture of GPT with MoE:

![GPT+MOE](images/moe/moe.png)

## MoE Architecture & Internals

If you want to understand how Mixture-of-Experts (MoE) models work and how they scale large language models efficiently, check out our comprehensive guide:

📖 **[MoE Internals Guide](MOE_GPT_INTERNALS.md)** - Complete breakdown of MoE components including:
- Expert Layers & Feed-Forward Networks
- Routing Mechanisms & Gating Functions
- Expert Capacity & Load Balancing
- Active Parameters vs Total Parameters
- Load Balancing Loss & Training Strategies
- MoE Layer Integration in Transformer Blocks

This guide includes code implementations, mathematical explanations, and visual diagrams to help you understand how MoE models achieve efficiency while maintaining performance in large-scale language models.

## Installation

This project uses `uv` for fast Python package management. To get started:

1. Install `uv` if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install the project dependencies:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```

To train your moe model, you can run following command. Data sample of shakespeare.txt is given for now under ```/gpe-moe/data/train_data/shakespeare.txt``` . You can change your data source based on reqs:

```bash
uv run python gpt-moe/train/gpt_train.py
``` 
To load weights from trained models you can use following command


## Dense-GPT Architecture
Following picture shows architecture of GPT

![GPT](images/gpt/gpt.png)

If you want to dive deep into the internals of GPT architecture and understand how each component works, check out our detailed guide:

📖 **[GPT Internals Guide](DENSE_GPT_INTERNALS.md)** - Complete breakdown of GPT components including:
- Token & Positional Embeddings
- Multi-Head Self-Attention mechanisms
- Feed-Forward Networks (MLP)
- Residual Connections & Layer Normalization
- Stacked Transformer Blocks
- Output Projection & Text Generation

This guide includes code implementations, mathematical explanations, and visual diagrams to help you understand every aspect of the GPT architecture from the ground up.


To train your model, you can run following command. Data sample of shakespeare.txt is given for now under ```/gpt/data/train_data/shakespeare.txt``` . You can change your data source based on reqs:

```bash
uv run python gpt/train/gpt_train.py
``` 
To load weights from trained models you can use following command

```bash
uv run python gpt/wight_loader/gpt2_weights_evaluator.py
```

To fine-tune the model, 2 fine-tuning examples is considered you can extend on that. First is for simple ham/spam classification fine tuner. Other is instruction following fine-tuner, again you can change the finetuning head and data based on your needs:

``` bash
uv run python gpt/fine_tuning/instruction_head_fine_tuning/instruction_head_find_tuner.py
```

```bash
uv run python gpt/fine_tuning/spam_head_fine_tuning/spam_head_fine_tuner.py
```
