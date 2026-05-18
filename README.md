# DA6401 Assignment 3: Implementing a Transformer for Machine Translation  
**Name:** Tanmay Gawande  
**Roll Number:** DA25M030  

## Links
- **GitHub Repository:** [Public repository link](https://github.com/Tanmay20030516/DA6401-assignment-3)  
- **W&B Report:** [Public report link](https://wandb.ai/da25m030-tanmay-gawande/da6401_assignment3/reports/DA6401-Assignment-3-Implementing-a-Transformer-for-Machine-Translation--VmlldzoxNjg4NDAzMg?accessToken=tlx69ays1pmu661j4rp3w6zfcnl9thc249tl1a07wg5fllffp0rscod7ua6f0aum)  

## Project Structure
```
.
├── dataset.py
├── model.py
├── train.py
├── lr_scheduler.py
└── requirements.txt
```

## Overview

A from-scratch PyTorch implementation of the Transformer architecture from ["Attention Is All You Need"](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) for German to English neural machine translation on the [Multi30k](https://huggingface.co/datasets/bentrevett/multi30k) dataset (29k train / 1k val / 1k test pairs).  


## Hyperparameters

| Parameter | Value |
|---|---|
| `d_model` | 256 |
| `N` (layers) | 4 |
| `num_heads` | 8 |
| `d_ff` | 1024 |
| `dropout` | 0.2 |
| `warmup_steps` | 2000 |
| `batch_size` | 128 |
| `epochs` | 60 |
| `label_smoothing` | 0.1 |
| `optimizer` | Adam (β₁=0.9, β₂=0.98, ε=1e-9) |

> Checkpoints saved based on best validation BLEU score.  

## Setup
```bash
pip install -r requirements.txt
python3 -m spacy download de_core_news_sm
python3 -m spacy download en_core_web_sm
```

## Training
```bash
# Full training pipeline with W&B logging
python3 train.py
```
This runs the complete experiment: dataset loading, vocabulary construction, model training with Noam scheduling, checkpoint saving, and final BLEU evaluation on the test set.  

## W&B Experiments

The report documents five ablation studies:  

1. **Noam Scheduler vs Fixed LR** — training loss and validation loss curves comparing warmup scheduling against a constant learning rate.  
2. **Scaling Factor Ablation** — with vs without 1/√dₖ, gradient norm analysis over the first 1000 steps.  
3. **Attention Rollout & Head Specialization** — per-head attention heatmaps from the last encoder layer.  
4. **Sinusoidal vs Learned Positional Encoding** — validation BLEU comparison.  
5. **Label Smoothing** — ε=0.1 vs ε=0.0, prediction confidence analysis.  

## Inference
```bash
# Evaluate BLEU on the test set using the best checkpoint
python3 -c "
from train import evaluate_bleu, load_checkpoint
from model import Transformer
from dataset import Multi30kDataset
import torch

m30k = Multi30kDataset()
m30k.build_vocab()
m30k.process_data()
test_loader = m30k.get_dataloader('test', batch_size=128)

model = Transformer(checkpoint_path="best_checkpoint.pth")
model.eval()

bleu = evaluate_bleu(model, test_loader, m30k.tgt_vocab)
print(f'Test BLEU: {bleu:.2f}')
"
```