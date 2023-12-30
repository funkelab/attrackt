import logging
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from attrackt.scripts.logger.logger import CSVLogger
from attrackt.scripts.trackastra.general_utils import (
    MLP,
    WarmupCosineLRScheduler,
    common_unsupervised_step,
)
from attrackt.trackastra.data import collate_sequence_padding
from attrackt.trackastra.model import TrackingTransformer

today_str = datetime.today().strftime("%Y-%m-%d")
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True


def train(
    datasets: dict[str, ConcatDataset],
    model: TrackingTransformer,
    batch_size: int,
    learning_rate: float,
    device: str,
    num_iterations: int,
    num_workers: int,
    causal_norm: str,
    lambda_: float,
    window: int,
    d_model: int,
    log_loss_every: int,
    suffix: str | None = None,
    warmup_iterations: int | None = None,
    include_transitive_loss: bool = True,
):
    # warmup iterations
    if not warmup_iterations:
        warmup_iterations = 10_000

    # initialize dataloaders
    sampler = torch.utils.data.RandomSampler(
        datasets["train"],
        num_samples=len(datasets["train"]),
        replacement=True,
    )

    train_dataloader = DataLoader(
        datasets["train"],
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_sequence_padding(),
    )

    val_dataloader = DataLoader(
        datasets["val"],
        batch_size=1,
        shuffle=False,
        num_workers=0 if num_workers == 0 else max(1, num_workers // 2),
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_sequence_padding(),
    )

    # initialize MLPs
    MLP_E = MLP(d_model=d_model).to(device)
    MLP_D = MLP(d_model=d_model).to(device)

    # initialize optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(MLP_E.parameters()) + list(MLP_D.parameters()),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    scheduler = WarmupCosineLRScheduler(optimizer, warmup_iterations, num_iterations)

    if suffix:
        experiment_dir = Path(f"fully-unsupervised-{today_str}-{suffix}")
    else:
        experiment_dir = Path(f"fully-unsupervised-{today_str}")

    model_dir = experiment_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    logger_data = CSVLogger(
        keys=[
            "iteration",
            "train_attrackt_loss",
            "train_transitive_loss",
            "train_loss",
            "val_loss",
        ],
        title=experiment_dir / "fully_unsupervised",
    )

    # model
    model = model.to(device)

    train_dataloader_iter = iter(train_dataloader)
    start_iteration = 0
    train_loss_average = 0.0
    min_loss = float("inf")

    total_params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_MLP_E_model = sum(
        p.numel() for p in MLP_E.parameters() if p.requires_grad
    )
    total_params_MLP_D_model = sum(
        p.numel() for p in MLP_D.parameters() if p.requires_grad
    )

    logger.info(
        f"Total trainable parameters: {total_params_model}, {total_params_MLP_E_model}, {total_params_MLP_D_model}"
    )
    num_updates = 0
    for iteration in range(start_iteration, num_iterations):
        try:
            train_batch = next(train_dataloader_iter)
        except StopIteration:
            train_dataloader_iter = iter(train_dataloader)
            train_batch = next(train_dataloader_iter)

        optimizer.zero_grad(set_to_none=True)
        train_attrackt_loss, train_transitive_loss, train_loss = (
            common_unsupervised_step(
                train_batch,
                model,
                MLP_E,
                MLP_D,
                device,
                causal_norm,
                window,
                lambda_,
                include_transitive_loss,
            )
        )
        if torch.isnan(train_loss):
            logger.info(
                f"At iteration {iteration}, NaN train loss. Skipping! train attrackt loss {train_attrackt_loss}. train transitive loss {train_transitive_loss}"
            )
            logger_data.add("iteration", iteration)
            logger_data.add("train_attrackt_loss", "")
            logger_data.add("train_transitive_loss", "")
            logger_data.add("train_loss", "")
        else:
            num_updates += 1
            logger_data.add("iteration", iteration)
            logger_data.add("train_attrackt_loss", train_attrackt_loss.item())
            logger_data.add("train_transitive_loss", train_transitive_loss.item())
            logger_data.add("train_loss", train_loss.item())
            train_loss_average += train_loss.item()
            train_loss.backward()
            optimizer.step()
            scheduler.step()
        if (iteration + 1) % log_loss_every == 0:
            train_loss_average /= num_updates
            logger.info(f"[Iter {iteration}] Avg Train Loss: {train_loss_average:.4f}")
            num_updates = 0
            model.eval()
            val_loss_average = 0.0
            num_updates_val = 0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_attrackt_loss, val_transitive_loss, val_loss = (
                        common_unsupervised_step(
                            val_batch,
                            model,
                            MLP_E,
                            MLP_D,
                            device,
                            causal_norm,
                            window,
                            lambda_,
                            include_transitive_loss,
                        )
                    )
                    if torch.isnan(val_loss):
                        logger.info(
                            f"At iteration {iteration}, NaN val loss. Skipping! val attrackt loss {val_attrackt_loss}. val transitive loss {val_transitive_loss}"
                        )
                    else:
                        num_updates_val += 1
                        val_loss_average += val_loss.item()
            val_loss_average /= num_updates_val
            logger.info(f"[Iter {iteration}] Avg Val Loss: {val_loss_average:.4f}")
            logger_data.add("val_loss", val_loss_average)
            checkpoint_data = {
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "train_loss_average": train_loss_average,
                "logger_data": logger_data.data,
                "scheduler_state_dict": scheduler.state_dict(),
            }

            (model_dir / "last").mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_data, model_dir / "last" / "last.pth")
            torch.save(
                MLP_E.state_dict(),
                model_dir / "last" / "MLP_E_weights_last.pth",
            )
            torch.save(
                MLP_D.state_dict(),
                model_dir / "last" / "MLP_D_weights_last.pth",
            )

            if val_loss_average < min_loss:
                min_loss = val_loss_average

                (model_dir / "best").mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint_data, model_dir / "best" / "best.pth")
                torch.save(
                    MLP_E.state_dict(),
                    model_dir / "best" / "MLP_E_weights.pth",
                )
                torch.save(
                    MLP_D.state_dict(),
                    model_dir / "best" / "MLP_D_weights.pth",
                )

                logger.info(f"New best model saved at iteration {iteration}")

            train_loss_average = 0.0
            model.train()
        else:
            logger_data.add("val_loss", "")
        logger_data.write(reset=True)
