import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
import numpy as np
from loguru import logger
import os

EMBED_PAD_NUM = -10000

def kl_loss_embedding_offline(logits, batch, tokenizer, args, label_probs, knn_label, query_embedding):
    # Pad labels
    shift_labels = batch['labels'][:, 1:].contiguous() # (batch, seq_len-1)
    nonpad_mask = shift_labels != -100
    shift_labels = shift_labels[nonpad_mask] # (nonpad b*t)

    # Pad logits
    nonpad_mask = torch.any(query_embedding != EMBED_PAD_NUM, dim=-1)
    shift_logits = logits[nonpad_mask] # (nonpad b*t, vocab_size)

    # Alignment sanity check
    assert shift_logits.shape == label_probs.shape, f"shift_logits.shape = {shift_logits.shape}, label_probs.shape = {label_probs.shape}"
    assert torch.all(shift_labels == knn_label), f"shift_labels and knn_label are not the same"

    # MLP Memory loss
    loss_fct = nn.CrossEntropyLoss()
    loss_ce = loss_fct(shift_logits, shift_labels)
    loss_kl = F.kl_div(F.log_softmax(shift_logits, dim=-1), label_probs, reduction='batchmean')

    total_loss = args.alpha * loss_kl + (1 - args.alpha) * loss_ce
    logger.info(f"KL loss weight: {args.alpha} KL loss: {loss_kl} CE loss: {loss_ce} Total loss: {total_loss}")

    return total_loss, loss_kl, loss_ce

def kl_loss_embedding_online(logits, batch, tokenizer, args, label_probs, knn_label):
    # Pad labels
    shift_labels = batch['labels'][:, 1:].contiguous() # (batch, seq_len-1)
    nonpad_mask = shift_labels != -100
    shift_labels = shift_labels[nonpad_mask] # (nonpad b*t)
    shift_logits = logits

    # Alignment sanity check
    assert shift_logits.shape == label_probs.shape, f"shift_logits.shape = {shift_logits.shape}, label_probs.shape = {label_probs.shape}"
    assert torch.all(shift_labels == knn_label), f"shift_labels and knn_label are not the same"

    # MLP Memory loss
    loss_fct = nn.CrossEntropyLoss()
    loss_ce = loss_fct(shift_logits, shift_labels)
    loss_kl = F.kl_div(F.log_softmax(shift_logits, dim=-1), label_probs, reduction='batchmean')

    total_loss = args.alpha * loss_kl + (1 - args.alpha) * loss_ce
    logger.info(f"KL loss weight: {args.alpha} KL loss: {loss_kl} CE loss: {loss_ce} Total loss: {total_loss}")

    return total_loss, loss_kl, loss_ce