from __future__ import annotations

import copy
import math
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_experiment_args, add_management_args, add_rehearsal_args


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Attention Knowledge Distillation with Pseudo-Bag Memory Pool.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Weight for attention distillation (L_attn).')
    parser.add_argument('--beta', type=float, default=0.2,
                        help='Weight for logits distillation (L_logits).')
    parser.add_argument('--pmp_ratio', type=float, default=0.2,
                        help='Fraction of patches kept when distilling a pseudo-bag.')
    parser.add_argument('--akd_temperature', type=float, default=1.0,
                        help='Temperature for KL distillation.')
    return parser


class PseudoBagPool:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.storage = []
        self.num_seen = 0

    def __len__(self) -> int:
        return len(self.storage)

    def is_empty(self) -> bool:
        return len(self.storage) == 0

    # 【修改 1】新增 attn 與 logits 儲存 (Algorithm 1: Line 19, 20)
    def add(self, bag0: torch.Tensor, bag1: torch.Tensor, label: torch.Tensor, 
            attn: torch.Tensor, logits: torch.Tensor) -> None:
        item = (
            bag0.detach().cpu().contiguous(),
            bag1.detach().cpu().contiguous(),
            label.detach().cpu().long().view(-1)[0],
            attn.detach().cpu().contiguous(),
            logits.detach().cpu().contiguous()
        )
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            idx = random.randint(0, self.num_seen)
            if idx < self.capacity:
                self.storage[idx] = item
        self.num_seen += 1

    def sample(self):
        if self.is_empty():
            return None
        bag0, bag1, label, attn, logits = random.choice(self.storage)
        return (bag0.to(self.device), bag1.to(self.device), label.to(self.device), 
                attn.to(self.device), logits.to(self.device))


class AkdPmp(ContinualModel):
    NAME = 'akd_pmp'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.memory = PseudoBagPool(args.buffer_size, self.device)
        self.current_task = 0
        # 【修改 2】移除 self.teacher，因為演算法依賴歷史緩存而非凍結模型

    def begin_task(self, dataset):
        self.net.train()

    def _distill_kl(self, student_preds: torch.Tensor, historical_preds: torch.Tensor) -> torch.Tensor:
        """通用的 KL 散度計算，適用於 Attention 與 Logits (Algorithm 1: Line 12, 13)"""
        if student_preds.dim() == 1:
            student_preds = student_preds.unsqueeze(0)
        if historical_preds.dim() == 1:
            historical_preds = historical_preds.unsqueeze(0)
            
        temp = max(float(self.args.akd_temperature), 1e-6)
        student_log = F.log_softmax(student_preds / temp, dim=-1)
        history_prob = F.softmax(historical_preds / temp, dim=-1)
        return F.kl_div(student_log, history_prob, reduction='batchmean') * (temp ** 2)

    def observe(self, inputs0, inputs1, labels, task, ssl=False):
        if ssl:
            return 0.0

        self.opt.zero_grad()
        
        # --- Algorithm 1: Line 7-9 (當前任務資料計算) ---
        outputs = self.net([inputs0, inputs1])
        logits = outputs[0]
        # 當前任務的 Cross-Entropy Loss
        loss = self.loss(logits, labels)

        # --- Algorithm 1: Line 10-13 (記憶池資料計算與蒸餾) ---
        if not self.memory.is_empty():
            replay = self.memory.sample()
            if replay is not None:
                buf0, buf1, buf_label, old_attn, old_logits = replay
                
                # 取得當前模型對舊資料的預測
                buf_outputs = self.net([buf0, buf1])
                buf_logits = buf_outputs[0]
                buf_attn = buf_outputs[-2]
                
                # 舊資料也要算標準的 CE Loss (這包含在 Line 9 的 D_bar 中)
                loss += self.loss(buf_logits, buf_label.view(1))
                
                # 計算 L_attn 與 L_logits 的 KL 散度
                l_attn = self._distill_kl(buf_attn, old_attn)
                l_logits = self._distill_kl(buf_logits, old_logits)
                
                # Algorithm 1: Line 17 (合併 Loss)
                loss = loss + self.args.alpha * l_attn + self.args.beta * l_logits

        loss.backward()
        self.opt.step()
        return float(loss.item())

    def _build_pseudo_bag(self, inputs0: torch.Tensor, inputs1: torch.Tensor, attn: torch.Tensor):
        bag_size = inputs0.shape[0]
        keep = max(1, int(math.ceil(self.args.pmp_ratio * bag_size)))
        
        if keep >= bag_size:
            idx = np.arange(bag_size)
        else:
            # 修正變數名稱，避免覆蓋原本的 tensor
            attn_np = attn.detach().flatten().cpu().numpy()
            top_k = max(1, keep // 4)
            bottom_k = max(1, keep // 4)
            rand_k = max(1, keep - top_k - bottom_k)
            
            while top_k + bottom_k + rand_k > keep:
                if rand_k > 1: rand_k -= 1
                elif top_k > 1: top_k -= 1
                elif bottom_k > 1: bottom_k -= 1
                else: break
            while top_k + bottom_k + rand_k < keep:
                rand_k += 1

            order = np.argsort(attn_np)
            low_idx = order[:bottom_k]
            high_idx = order[-top_k:]
            remaining = np.setdiff1d(np.arange(bag_size), np.union1d(low_idx, high_idx))
            
            if remaining.size == 0:
                rand_idx = np.array([], dtype=int)
            else:
                replace = remaining.size < rand_k
                rand_idx = np.random.choice(remaining, size=rand_k, replace=replace)
                
            idx = np.unique(np.concatenate([low_idx, high_idx, rand_idx]))
            if idx.size == 0:
                idx = np.array([int(np.argmax(attn_np))])
            idx.sort()

        pseudo0 = inputs0[idx].detach().contiguous()
        pseudo1 = inputs1[idx].detach().contiguous()
        
        # 【修改 3】同時萃取並保存對應的 Attention 分數 (確保維度一致)
        pseudo_attn = attn.flatten()[idx].unsqueeze(0).detach().contiguous()
        
        return pseudo0, pseudo1, pseudo_attn

    def save_buffer(self, inputs0, inputs1, labels, task):
        if inputs0.shape[0] <= 0:
            return
            
        with torch.no_grad():
            outputs = self.net([inputs0, inputs1])
            
        logits = outputs[0]
        attn = outputs[-2]
        
        # Algorithm 1: Line 19 (Distill)
        pseudo0, pseudo1, pseudo_attn = self._build_pseudo_bag(inputs0, inputs1, attn)
        
        # Algorithm 1: Line 20 (儲存進 reservoir)
        self.memory.add(pseudo0, pseudo1, labels, pseudo_attn, logits)

    def end_task(self, dataset):
        # 移除了 Teacher 模型的複製邏輯
        self.current_task += 1
