import logging
import warnings

import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler

from attrackt.trackastra.model import TrackingTransformer
from attrackt.trackastra.utils import blockwise_causal_norm

logger = logging.getLogger(__name__)


class CustomWrapperForPEFT(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(
        self, input_ids=None, coords=None, features=None, padding_mask=None, **kwargs
    ):
        if coords is None or features is None:
            raise ValueError("Coordinates and features must be provided.")
        return self.base_model(coords, features, padding_mask)

    def save(self, x):
        self.base_model.save(x)


class MLP(torch.nn.Module):
    def __init__(self, d_model: int, expand: float = 2, bias: bool = True):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_model, int(d_model * expand))
        self.fc2 = torch.nn.Linear(int(d_model * expand), d_model, bias=bias)
        self.act = torch.nn.GELU()
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class WarmupCosineLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_iterations,
        num_iterations,
        cosine_final: float = 0.001,
        last_iteration=-1,
        verbose=False,
    ):
        self.warmup_iterations = warmup_iterations
        self.num_iterations = num_iterations
        self.cosine_final = cosine_final
        super().__init__(optimizer, last_iteration)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        if self.last_epoch < self.warmup_iterations:
            # linear ramp
            initial = 1e-2
            factor = initial + (1 - initial) * self.last_epoch / self.warmup_iterations
        else:
            iteration_rel = (self.last_epoch - self.warmup_iterations) / (
                self.num_iterations - self.warmup_iterations + 1
            )
            factor = (
                0.5 * (1 + np.cos(np.pi * iteration_rel)) * (1 - self.cosine_final)
                + self.cosine_final
            )

        # logging.info(f"LRScheduler: relative lr factor {factor:.03f}")
        return [factor * base_lr for base_lr in self.base_lrs]


def compute_symmetric_loss(
    A_soft: torch.Tensor,
    mask_invalid: torch.Tensor | None = None,
    ignore_diagonal: bool = True,
) -> torch.Tensor:
    diff = A_soft - A_soft.transpose(1, 2)  # compare within each sample only
    diff2 = diff.pow(2)

    # Build valid mask
    if mask_invalid is not None:
        valid = ~mask_invalid
    else:
        valid = torch.ones_like(diff2, dtype=torch.bool)

    if ignore_diagonal:
        B, N, _ = A_soft.shape
        eye = torch.eye(N, dtype=torch.bool, device=A_soft.device).unsqueeze(
            0
        )  # (1,N,N)
        valid = valid & ~eye  # drop diagonal per sample

    # Zero out invalids; average per sample to avoid cross-sample mixing
    diff2 = diff2.masked_fill(~valid, 0.0)
    per_sample_sum = diff2.sum(dim=(1, 2))
    per_sample_count = valid.sum(dim=(1, 2)).clamp_min(1)  # avoid /0
    per_sample_mean = per_sample_sum / per_sample_count  # (B,)

    return per_sample_mean.mean()  # scalar


def compute_regularization_loss(
    A: torch.tensor,
    indices,
    mse_loss,
    max_step,
):
    loss = torch.tensor(0.0, requires_grad=True, device=A.device)
    counter = 0
    for step in range(0, max_step - 1):
        if step in indices and step + 1 in indices and step + 2 in indices:
            a = A[indices[step]][:, indices[step + 1]]
            b = A[indices[step + 1]][:, indices[step + 2]]
            c = A[indices[step]][:, indices[step + 2]]
            loss = loss + mse_loss(c, torch.matmul(a, b))
            counter += 1
    if counter == 0:
        return loss
    else:
        return loss / counter


def compute_transitive_loss(timepoints, A_soft, window, mse_loss_fn, device):
    transitive_loss = torch.tensor(0.0, requires_grad=True).to(device)
    for b in range(timepoints.shape[0]):
        min_t = torch.min(timepoints[b])
        indices = {
            t: (timepoints[b] == min_t + t).nonzero(as_tuple=True)[0]
            for t in range(window)
        }

        indices = {t: idx for t, idx in indices.items() if idx.numel() > 0}

        transitive_loss = transitive_loss + compute_regularization_loss(
            A_soft[b],
            indices,
            mse_loss_fn,
            max_step=window - 1,
        )

    return transitive_loss


def common_supervised_step(
    batch: dict,
    model: TrackingTransformer,
    MLP_E: torch.nn.Module,
    MLP_D: torch.nn.Module,
    device: str,
    causal_norm: str,
    window: int,
    delta_cutoff: int = 2,
    eps: float = 1e-8,
    lambda_: float = 0.1,
    include_transitive_loss: bool = False,
):
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    criterion_softmax = torch.nn.BCELoss(reduction="none")
    timepoints = batch["timepoints"].to(device)
    coords = batch["coords"].to(device)
    features = batch["features"].to(device)
    padding_mask = batch["padding_mask"].bool().to(device)
    A_GT = batch["assoc_matrix"].to(device)
    A_mask = batch["assoc_matrix_mask"].to(device)

    A = model(coords=coords, features=features, padding_mask=padding_mask)
    A.clamp_(-20, 20)
    mask_invalid = torch.logical_or(
        padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
    ).to(A.device)
    A[mask_invalid] = 0

    # compute final mask
    mask_valid = ~mask_invalid
    dt = timepoints.unsqueeze(1) - timepoints.unsqueeze(2)
    mask_time = torch.logical_and(dt > 0, dt <= delta_cutoff)
    mask = mask_valid * mask_time * A_mask
    mask = mask.float()

    # compute un-reduced loss
    loss = criterion(A, A_GT)

    if causal_norm != "none":
        A_soft = torch.stack(
            [
                blockwise_causal_norm(_A, _t, mode=causal_norm, mask_invalid=_m)
                for _A, _t, _m in zip(A, timepoints, mask_invalid)
            ]
        )
        with torch.amp.autocast(device, enabled=False):
            loss = 0.01 * loss + criterion_softmax(A_soft, A_GT)
    else:
        A_soft = torch.sigmoid(A)

    with torch.no_grad():
        pos_region = (mask == 1) & (A_GT == 1)
        neg_region = (mask == 1) & (A_GT == 0)
        num_pos = pos_region.sum().float()
        num_neg = neg_region.sum().float()
        pos_weight = num_neg / (num_pos + eps)
        # logger.info(f"Weighting factor is {pos_weight}.")
        loss_weight = torch.ones_like(A, dtype=A.dtype)
        loss_weight[pos_region] = pos_weight
        loss_weight[mask == 0.0] = 0.0

    weighted = loss * loss_weight

    # Sum over (1,2), keep batch dim
    weight_sums = loss_weight.sum(dim=(1, 2)) + eps
    loss_per_sample = weighted.sum(dim=(1, 2)) / weight_sums

    has_any = weight_sums > eps
    if has_any.any():
        loss = loss_per_sample[has_any].mean()
    else:
        loss = torch.tensor(float("nan"), device=A.device)

    if include_transitive_loss and lambda_ > 0:
        timepoints[timepoints == -1] = 10_000_000
        transitive_loss = compute_transitive_loss(
            timepoints, A_soft, window, torch.nn.MSELoss(reduction="mean"), device
        )
        transitive_loss = lambda_ * transitive_loss
        loss = loss + transitive_loss

    return loss


def common_unsupervised_step(
    batch: dict,
    model: TrackingTransformer,
    MLP_E: torch.nn.Module,
    MLP_D: torch.nn.Module,
    device: str,
    causal_norm: str,
    window: int,
    lambda_: float,
    include_transitive_loss: bool = True,
):
    mse_loss_fn_not_reduced = torch.nn.MSELoss(reduction="none")
    mse_loss_fn_reduced = torch.nn.MSELoss(reduction="mean")
    autoencoder_features = batch["autoencoder_features"].to(device)
    timepoints = batch["timepoints"].to(device)
    coords = batch["coords"].to(device)
    features = batch["features"].to(device)
    padding_mask = batch["padding_mask"].bool().to(device)
    A = model(coords=coords, features=features, padding_mask=padding_mask)
    A.clamp_(-20, 20)
    mask_invalid = torch.logical_or(
        padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
    ).to(A.device)
    A[mask_invalid] = 0
    if causal_norm != "none":
        A_soft = torch.stack(
            [
                blockwise_causal_norm(_A, _t, mode=causal_norm, mask_invalid=_m)
                for _A, _t, _m in zip(A, timepoints, mask_invalid)
            ]
        )
    else:
        A_soft = torch.sigmoid(A)

    autoencoder_features_shape = autoencoder_features.shape
    post_MLP_E = MLP_E(autoencoder_features.view(-1, autoencoder_features_shape[-1]))
    post_MLP_E = post_MLP_E.view(autoencoder_features_shape)
    attrackt_loss = torch.tensor(0.0, requires_grad=True).to(device)
    timepoints[timepoints == -1] = 10_000_000
    timepoint_diff = timepoints.unsqueeze(2) - timepoints.unsqueeze(1)
    for dt in range(1, window):
        mask = timepoint_diff == -dt
        masked_A = A_soft * mask  # (b, N, N)
        features_after_attention = torch.bmm(masked_A.transpose(1, 2), post_MLP_E)
        post_MLP_D = MLP_D(
            features_after_attention.view(-1, features_after_attention.shape[-1])
        )
        post_MLP_D = post_MLP_D.view(autoencoder_features_shape)
        loss_not_reduced = mse_loss_fn_not_reduced(post_MLP_D, autoencoder_features)
        valid_rows_mask = ~(
            torch.all(features_after_attention == 0, dim=2)  # Rows that are all zeros
        )
        valid_elements_mask = valid_rows_mask.unsqueeze(-1).expand_as(loss_not_reduced)
        valid_loss = loss_not_reduced[valid_elements_mask]
        attrackt_loss = attrackt_loss + valid_loss.mean()

    if include_transitive_loss and lambda_ > 0:
        transitive_loss = compute_transitive_loss(
            timepoints, A_soft, window, mse_loss_fn_reduced, device
        )
        transitive_loss = lambda_ * transitive_loss
        loss = attrackt_loss + transitive_loss
        return attrackt_loss, transitive_loss, loss
    else:
        return attrackt_loss, torch.tensor(0.0).to(device), attrackt_loss
