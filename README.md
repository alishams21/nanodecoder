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
   | Wandb logging | ✅ | ✅ | ✅ | ✅ |


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

A GPT-style model that replaces certain dense feedforward layers with Mixture-of-Experts (MoE) layers. Instead of activating all parameters for every token, only a subset of specialized experts are used, making the model more efficient while retaining high capacity. Examples include GPT-OSS, DeepSeek-V3, and Switch Transformer. Following image shows the architecture of MoE-GPT:

![GPT+MOE](images/moe/moe.png)

**For Production and MoE-GPT Internals check out**: 
📖 **[Production + MoE Internals Guide](PRODUCTION_MOE_GPT_INTERNALS.md)** 


## Dense-GPT Architecture
In this architecture all parameters of model are active for every token. This design ensures the full network is always utilized during training and inference, making it straightforward to implement and analyze. Well-known examples include GPT-2, GPT-3, and GPT-4, which follow this dense architecture to deliver consistent performance across a wide range of tasks. Following picture shows architecture of DENSE-GPT:

![GPT](images/gpt/gpt.png)

**For detailed guide on DENSE-GPT** 📖 **[GPT Internals Guide](DENSE_GPT_INTERNALS.md)** 

