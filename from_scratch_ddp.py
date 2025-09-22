"""
Distributed Data Parallel (DDP) training script for GPT models from scratch.
Adds multi-GPU DDP support on top of mixed precision version.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration - DDP enabled
# I/O
out_dir = 'out_ddp'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # Simulate larger batch sizes
batch_size = 12  # Micro-batch size per GPU
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# moe
n_exp = 1
top_k = 2
use_aux_loss = False
use_router_z_loss = False
use_noisy_top_k = False
aux_loss_weight = 0.001
router_z_loss_weight = 0.01
train_capacity = 1.25
eval_capacity = 2.0
min_capacity = 4
stride = 2
use_switch_tfm_init = False
switch_tfm_init_scale = 1.0
router_use_full_prec = False

# optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# DDP settings
backend = 'nccl'

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# -----------------------------------------------------------------------------
# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    # Scale down gradient accumulation steps per process
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

if master_process:
    print("=" * 50)
    print("DDP TRAINING SCRIPT")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"Data type: {dtype}")
    print(f"Batch size per GPU: {batch_size}")
    print(f"Total batch size: {batch_size * ddp_world_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Tokens per iteration: {tokens_per_iter:,}")
    print(f"Block size: {block_size}")
    print(f"Model size: {n_layer} layers, {n_head} heads, {n_embd} embd")
    print(f"DDP world size: {ddp_world_size}")
    print("=" * 50)

# setup
if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)

# enable optimizations for GPU
if device == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'

# mixed precision context
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loader with GPU support
data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # Move to GPU with pin_memory for faster transfer
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init training state
iter_num = 0
best_val_loss = 1e9

# get vocab size
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    if master_process:
        print(f"Found vocab_size = {meta_vocab_size}")

# model init
if master_process:
    print("Initializing model from scratch...")
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, n_exp=n_exp, top_k=top_k,
                  use_aux_loss=use_aux_loss, use_router_z_loss=use_router_z_loss,
                  use_noisy_top_k=use_noisy_top_k, aux_loss_weight=aux_loss_weight,
                  router_z_loss_weight=router_z_loss_weight, train_capacity=train_capacity,
                  eval_capacity=eval_capacity, min_capacity=min_capacity, stride=stride,
                  use_switch_tfm_init=use_switch_tfm_init, switch_tfm_init_scale=switch_tfm_init_scale,
                  router_use_full_prec=router_use_full_prec)

model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304

gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)
if master_process:
    print(f"Model created with {model.get_num_params()/1e6:.2f}M parameters")

# initialize GradScaler for mixed precision
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# wrap model in DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# loss estimation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# training loop
if master_process:
    print("Starting training...")
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    # set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"Saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    if iter_num == 0 and eval_only:
        break

    # forward backward update with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # Only sync gradients at the last micro step
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
        
        # Async prefetch next batch while GPU is busy
        X, Y = get_batch('train')
        
        # backward pass with gradient scaling
        scaler.scale(loss).backward()
    
    # gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # optimizer step with scaler
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # logging
    if iter_num % log_interval == 0 and master_process:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"Iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    
    iter_num += 1
    local_iter_num += 1

    # termination
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

if master_process:
    print("Training completed!")
