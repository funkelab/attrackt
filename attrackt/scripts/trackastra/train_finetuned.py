import logging
import os
import re
from datetime import datetime
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import ConcatDataset, DataLoader

from attrackt.scripts.logger.logger import CSVLogger
from attrackt.scripts.trackastra.general_utils import (
    MLP,
    CustomWrapperForPEFT,
    WarmupCosineLRScheduler,
    common_supervised_step,
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
    checkpoint_path: str,
    batch_size: int,
    learning_rate: float,
    device: str,
    num_iterations: int,
    num_workers: int,
    causal_norm: str,
    lambda_: float,
    window: int,
    d_model: int,
    do_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    log_loss_every: int = 1000,
    eps: float = 1e-8,
    include_transitive_loss_supervised: bool = False,
    suffix: str | None = None,
):
    warmup_iterations = 0  # no need for warmup in fine-tuning script.

    # initialize dataloaders
    unsupervised_sampler = torch.utils.data.RandomSampler(
        datasets["train_unsupervised"],
        num_samples=len(datasets["train_unsupervised"]),
        replacement=True,
    )

    train_unsupervised_dataloader = DataLoader(
        datasets["train_unsupervised"],
        sampler=unsupervised_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_sequence_padding(),
    )

    supervised_sampler = torch.utils.data.RandomSampler(
        datasets["train_supervised"],
        num_samples=len(datasets["train_supervised"]),
        replacement=True,
    )

    train_supervised_dataloader = DataLoader(
        datasets["train_supervised"],
        sampler=supervised_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_sequence_padding(),
    )

    val_supervised_dataloader = DataLoader(
        datasets["val_supervised"],
        batch_size=1,
        shuffle=False,
        num_workers=0 if num_workers == 0 else max(1, num_workers // 2),
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_sequence_padding(),
    )

    pseudo_supervised_sampler = torch.utils.data.RandomSampler(
        datasets["train_pseudo_supervised"],
        num_samples=len(datasets["train_pseudo_supervised"]),
        replacement=True,
    )

    train_pseudo_supervised_dataloader = DataLoader(
        datasets["train_pseudo_supervised"],
        sampler=pseudo_supervised_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_sequence_padding(),
    )

    val_pseudo_supervised_dataloader = DataLoader(
        datasets["val_pseudo_supervised"],
        batch_size=1,
        shuffle=False,
        num_workers=0 if num_workers == 0 else max(1, num_workers // 2),
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_sequence_padding(),
    )

    # load weights from checkpoint.
    model = model.to(device)
    MLP_E = MLP(d_model=d_model).to(device)
    MLP_D = MLP(d_model=d_model).to(device)

    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device(device), weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Successfully loaded pre-trained model weights")

    checkpoint_MLP_E = torch.load(
        os.path.dirname(checkpoint_path) + "/MLP_E_weights.pth", map_location=device
    )
    MLP_E.load_state_dict(checkpoint_MLP_E)

    checkpoint_MLP_D = torch.load(
        os.path.dirname(checkpoint_path) + "/MLP_D_weights.pth", map_location=device
    )
    MLP_D.load_state_dict(checkpoint_MLP_D)

    logger.info("Successfully loaded MLP_E and MLP_D")

    if do_lora:
        # wrap model using CustomWrapper.
        model = CustomWrapperForPEFT(model)
        target_modules = [
            name
            for name, module in model.named_modules()
            if re.search(r"attn\.(q_pro|k_pro|v_pro|proj)", name)
        ]
        logger.info(f"Target modules are {target_modules}.")
        # specify a LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        model = get_peft_model(model, lora_config)

    # initialize optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(MLP_E.parameters()) + list(MLP_D.parameters()),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    scheduler = WarmupCosineLRScheduler(optimizer, warmup_iterations, num_iterations)

    if suffix:
        experiment_dir = Path(f"finetuned-{today_str}-{suffix}")
    else:
        experiment_dir = Path(f"finetuned-{today_str}")
    model_dir = experiment_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    logger_data = CSVLogger(
        keys=[
            "iteration",
            "train_supervised_loss",
            "train_pseudo_supervised_loss",
            "train_unsupervised_loss",
            "train_loss",
            "val_loss",
        ],
        title=experiment_dir / "finetuned",
    )

    train_supervised_dataloader_iter = iter(train_supervised_dataloader)
    train_pseudo_supervised_dataloader_iter = iter(train_pseudo_supervised_dataloader)
    train_unsupervised_dataloader_iter = iter(train_unsupervised_dataloader)

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
            train_supervised_batch = next(train_supervised_dataloader_iter)
        except StopIteration:
            train_supervised_dataloader_iter = iter(train_supervised_dataloader)
            train_supervised_batch = next(train_supervised_dataloader_iter)

        try:
            train_pseudo_supervised_batch = next(
                train_pseudo_supervised_dataloader_iter
            )
        except StopIteration:
            train_pseudo_supervised_dataloader_iter = iter(
                train_pseudo_supervised_dataloader
            )
            train_pseudo_supervised_batch = next(
                train_pseudo_supervised_dataloader_iter
            )

        try:
            train_unsupervised_batch = next(train_unsupervised_dataloader_iter)
        except StopIteration:
            train_unsupervised_dataloader_iter = iter(train_unsupervised_dataloader)
            train_unsupervised_batch = next(train_unsupervised_dataloader_iter)

        optimizer.zero_grad(set_to_none=True)

        train_supervised_loss = common_supervised_step(
            train_supervised_batch,
            model,
            MLP_E,
            MLP_D,
            device,
            causal_norm,
            window,
            include_transitive_loss=include_transitive_loss_supervised,
        )

        train_pseudo_supervised_loss = common_supervised_step(
            train_pseudo_supervised_batch,
            model,
            MLP_E,
            MLP_D,
            device,
            causal_norm,
            window,
            include_transitive_loss=include_transitive_loss_supervised,
        )

        _, _, train_unsupervised_loss = common_unsupervised_step(
            train_unsupervised_batch,
            model,
            MLP_E,
            MLP_D,
            device,
            causal_norm,
            window,
            lambda_,
        )
        train_loss = (
            train_supervised_loss
            + train_pseudo_supervised_loss
            + train_unsupervised_loss
        )

        if torch.isnan(train_loss):
            logger.info(
                f"At iteration {iteration}, NaN train loss. Skipping! train supervised loss {train_supervised_loss.item()}. train pseudo supervised loss {train_pseudo_supervised_loss.item()} train unsupervised loss {train_unsupervised_loss.item()}"
            )
            logger_data.add("iteration", iteration)
            logger_data.add("train_supervised_loss", "")
            logger_data.add("train_pseudo_supervised_loss", "")
            logger_data.add("train_unsupervised_loss", "")
            logger_data.add("train_loss", "")
        else:
            num_updates += 1
            logger_data.add("iteration", iteration)
            logger_data.add("train_supervised_loss", train_supervised_loss.item())
            logger_data.add(
                "train_pseudo_supervised_loss", train_pseudo_supervised_loss.item()
            )
            logger_data.add("train_unsupervised_loss", train_unsupervised_loss.item())
            logger_data.add("train_loss", train_loss.item())
            train_loss_average += train_loss.item()
            train_loss.backward()
            optimizer.step()
            scheduler.step()
        if (iteration + 1) % log_loss_every == 0:
            train_loss_average /= num_updates + eps
            logger.info(f"[Iter {iteration}] Avg Train Loss: {train_loss_average:.4f}")
            num_updates = 0
            model.eval()
            val_loss_average = 0.0
            num_updates_val = 0
            with torch.no_grad():
                for (
                    val_supervised_batch,
                    val_pseudo_supervised_batch,
                ) in zip(
                    val_supervised_dataloader,
                    val_pseudo_supervised_dataloader,
                ):
                    val_supervised_loss = common_supervised_step(
                        val_supervised_batch,
                        model,
                        MLP_E,
                        MLP_D,
                        device,
                        causal_norm,
                        window,
                        include_transitive_loss=include_transitive_loss_supervised,
                    )
                    val_pseudo_supervised_loss = common_supervised_step(
                        val_pseudo_supervised_batch,
                        model,
                        MLP_E,
                        MLP_D,
                        device,
                        causal_norm,
                        window,
                        include_transitive_loss=include_transitive_loss_supervised,
                    )

                    val_loss = val_supervised_loss + val_pseudo_supervised_loss
                    if torch.isnan(val_loss):
                        logger.info(
                            f"At iteration {iteration}, NaN val loss. Skipping! val supervised loss {val_supervised_loss.item()}. val pseudo supervised loss {val_pseudo_supervised_loss.item()}"
                        )
                    else:
                        num_updates_val += 1
                        val_loss_average += val_loss.item()
            val_loss_average /= num_updates_val + eps
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
                model_dir / "last" / "MLP_E_weights.pth",
            )
            torch.save(
                MLP_D.state_dict(),
                model_dir / "last" / "MLP_D_weights.pth",
            )
            # saving adapter weights.
            if do_lora:
                model.save_pretrained(model_dir / "last")

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
                # saving adapter weights.
                if do_lora:
                    model.save_pretrained(model_dir / "best")

                logger.info(f"New best model saved at iteration {iteration}")

            train_loss_average = 0.0
            model.train()
        else:
            logger_data.add("val_loss", "")
        logger_data.write(reset=True)
