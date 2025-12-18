# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

#from dinov3.distributed import get_process_subgroup, get_subgroup_size
from src.training.utils.distributed import get_process_subgroup, get_subgroup_size

def lossfunc(t, s, temp):  # noqa: F811
    return torch.sum(t.float() * F.log_softmax(s.float() / temp, dim=-1), dim=-1)


class SinkhornKnoppTeacher(nn.Module):
    """
    Versione semplificata: single-GPU, niente n_masked_patches_tensor esplicito.
    """

    @torch.no_grad()
    def forward(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        Q = torch.exp(teacher_output / teacher_temp).t()  # [K, B]
        B = Q.shape[1]
        K = Q.shape[0]

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q, group=get_process_subgroup())
        Q /= sum_Q

        for _ in range(n_iterations):
            # normalize rows
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows, group=get_process_subgroup())
            Q /= sum_of_rows
            Q /= K

            # normalize columns
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B
        return Q.t()



class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.full((1, 1, patch_out_dim), math.nan))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None
        self.sinkhorn_knopp_teacher = SinkhornKnoppTeacher()

    def init_weights(self) -> None:
        self.center.zero_()

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp, update_centers=True):
        if update_centers:
            self.apply_center_update()
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks_flat):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_patch_tokens: (B, N, D) tensor
        teacher_patch_tokens: (B, N, D) tensor
        student_masks_flat: (B, N) tensor
        """
        t = teacher_patch_tokens
        s = student_patch_tokens
        loss = lossfunc(t, s, self.student_temp)
        loss = torch.sum(loss * student_masks_flat.float(), dim=-1) / student_masks_flat.sum(dim=-1).clamp(min=1.0)
        return -loss.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        """
        student_patch_tokens_masked: [B*T, K] oppure [B, T, K] flattenato fuori
        teacher_patch_tokens_masked: stessa shape di student
        student_masks_flat: bool [B*T] che indica quali patch sono mascherati
        """
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked

        # loss per patch (ancora non media): [B*T]
        loss = lossfunc(t, s, self.student_temp)

        # Seleziona solo i patch mascherati
        if student_masks_flat.dtype != torch.bool:
            student_masks_flat = student_masks_flat.bool()
        masked_loss = loss[student_masks_flat]

        if n_masked_patches is not None:
            masked_loss = masked_loss[:n_masked_patches]

        if masked_loss.numel() == 0:
            # nessun patch mascherato (caso limite): loss zero
            return loss.new_tensor(0.0)

        # media semplice sui patch mascherati (segno meno per avere cross-entropy)
        return -masked_loss.mean()


    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(teacher_patch_tokens.mean(1), dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True, group=get_process_subgroup())

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = get_subgroup_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True
