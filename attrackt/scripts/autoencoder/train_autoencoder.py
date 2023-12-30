import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import torch
import zarr
from torch.utils.data import DataLoader

from attrackt.autoencoder.autoencoder_model import AutoencoderModel
from attrackt.autoencoder.autoencoder_model3d import AutoencoderModel3d
from attrackt.autoencoder.zarr_csv_dataset_autoencoder import ZarrCsvDatasetAutoencoder
from attrackt.scripts.logger.logger import CSVLogger

today_str = datetime.today().strftime("%Y-%m-%d")
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True


def save_snapshots(
    output: torch.Tensor, data: torch.Tensor, iteration: int, snapshots_dir: Path
) -> None:
    """
    Save raw input and predicted output as Zarr snapshots.

    Parameters
    ----------
    output : torch.Tensor
        The predicted output tensor. Shape: (B, C, ...) where ... is spatial.
    data : torch.Tensor
        The raw input tensor. Shape: (B, C, ...) matching output shape.
    iteration : int
        Snapshot identifier for the Zarr group.
    """
    f = zarr.open(snapshots_dir / "snapshots.zarr", mode="a")

    num_spatial_dims = data.ndim - 2
    if num_spatial_dims == 2:
        axis_names = ["c", "t", "y", "x"]
    elif num_spatial_dims == 3:
        axis_names = ["c", "t", "z", "y", "x"]
    else:
        raise ValueError(
            f"Unsupported number of spatial dimensions: {num_spatial_dims}"
        )

    offset = [0] * (num_spatial_dims + 1)
    resolution = [1] * (num_spatial_dims + 1)

    def save_tensor(tensor: torch.Tensor, key: str) -> None:
        arr = (
            tensor.cpu()
            .detach()
            .numpy()
            .transpose(1, 0, *range(2, 2 + num_spatial_dims))
        )
        f[f"{iteration}/{key}"] = arr
        f[f"{iteration}/{key}"].attrs.update(
            {
                "axis_names": axis_names,
                "offset": offset,
                "resolution": resolution,
            }
        )

    save_tensor(data, "raw")
    save_tensor(output, "pred")

    logger.info(f"Saved snapshot for iteration {iteration}")


def train_autoencoder(
    dataset: ZarrCsvDatasetAutoencoder,
    dataset_val: ZarrCsvDatasetAutoencoder,
    model: Union["AutoencoderModel", "AutoencoderModel3d"],
    batch_size: int,
    num_iterations: int,
    learning_rate: float,
    device: str,
    num_workers: int = 0,
    log_loss_every: int = 100,
    save_snapshots_every: int = 10000,
    checkpoint: Optional[str] = None,
    suffix: str | None = None,
) -> None:
    """
    Train a 2D or 3D autoencoder with optional logging and checkpointing.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Training dataset.
    dataset_val : torch.utils.data.Dataset
        Validation dataset.
    model : AutoencoderModel | AutoencoderModel3d
        The model to train.
    batch_size : int
        Batch size.
    num_iterations : int
        Total number of training iterations.
    learning_rate : float
        Learning rate for optimizer.
    device : str
        Device string, e.g. "cuda" or "cpu".
    log_loss_every : int
        Frequency of logging and validation.
    save_snapshots_every : int
        Frequency of snapshot saving.
    checkpoint : Optional[str]
        Path to checkpoint file to resume from.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    dataloader_val = DataLoader(
        dataset_val, batch_size=batch_size, num_workers=num_workers
    )

    if model.variational:
        logvar_init = 0.0
        logvar = torch.nn.Parameter(torch.tensor(logvar_init, device=device))
        optimizer = torch.optim.Adam(
            list(model.parameters()) + [logvar], lr=learning_rate
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda it: (1 - it / num_iterations) ** 0.9
    )

    if suffix:
        experiment_dir = Path(f"autoencoder-{today_str}-{suffix}")
    else:
        experiment_dir = Path(f"autoencoder-{today_str}")
    model_dir = experiment_dir / "models_autoencoder"
    snapshots_dir = experiment_dir / "snapshots_autoencoder"
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    logger_data = CSVLogger(
        keys=[
            "iteration",
            "train_mse_loss",
            "train_nll_loss",
            "train_kl_loss",
            "train_loss",
            "val_loss",
        ],
        title=experiment_dir / "autoencoder",
    )

    start_iteration = 0
    loss_average = 0.0
    min_loss = float("inf")

    if checkpoint:
        logger.info(f"Resuming from checkpoint: {checkpoint}")
        state = torch.load(checkpoint, map_location=device)
        start_iteration = state["iteration"] + 1
        loss_average = state["loss_average"]
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optim_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        logger_data.data = state["logger_data"]

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params}")

    dataloader_iter = iter(dataloader)
    const = 0.5 * math.log(2.0 * math.pi)

    for iteration in range(start_iteration, num_iterations):
        try:
            img_crop, *_ = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            img_crop, *_ = next(dataloader_iter)
        img_crop = img_crop.to(device)

        optimizer.zero_grad()
        if model.variational:
            recon, posterior = model(img_crop)  # recon ~ mu_x
            mse = (img_crop - recon) ** 2
            logger_data.add(
                "train_mse_loss", mse.view(mse.size(0), -1).sum(dim=1).mean().item()
            )
            logvar_clamped = logvar.clamp(-8.0, 5.0)
            nll = 0.5 * (mse * torch.exp(-logvar_clamped) + logvar_clamped) + const
            # nll = 0.5 * (mse * torch.exp(-logvar) + logvar)  # broadcast scalar logvar
            recon_loss = nll.view(nll.size(0), -1).sum(dim=1).mean()
            logger_data.add("train_nll_loss", recon_loss.item())
            kl = posterior.kl()
            kl_loss = kl.view(kl.size(0), -1).sum(dim=1).mean()
            logger_data.add("train_kl_loss", kl_loss.item())
            beta = 0.000001
            loss = recon_loss + beta * kl_loss
        else:
            recon = model(img_crop)
            mse = (img_crop - recon) ** 2
            loss = mse.view(mse.size(0), -1).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        logger_data.add("iteration", iteration)
        logger_data.add("train_loss", loss.item())
        loss_average += loss.item()

        if (iteration + 1) % save_snapshots_every == 0:
            save_snapshots(recon, img_crop, iteration, snapshots_dir)

        if (iteration + 1) % log_loss_every == 0:
            loss_average /= log_loss_every
            logger.info(f"[Iter {iteration}] Avg Train Loss: {loss_average:.4f}")

            # Evaluate on validation set
            model.eval()
            val_loss = 0.0
            count = 0
            with torch.no_grad():
                for val_crop, *_ in dataloader_val:
                    val_crop = val_crop.to(device)
                    if model.variational:
                        val_recon, val_posterior = model(val_crop)
                        val_mse = (val_crop - val_recon) ** 2
                        val_logvar_clamped = logvar.clamp(-8.0, 5.0)
                        val_nll = (
                            0.5
                            * (
                                val_mse * torch.exp(-val_logvar_clamped)
                                + val_logvar_clamped
                            )
                            + const
                        )
                        val_recon_loss = (
                            val_nll.view(val_nll.size(0), -1).sum(dim=1).mean()
                        )
                        val_kl = val_posterior.kl()
                        val_kl_loss = val_kl.view(val_kl.size(0), -1).sum(dim=1).mean()
                        val_loss += (val_recon_loss + beta * val_kl_loss).item()
                    else:
                        val_recon = model(val_crop)
                        val_mse = (val_crop - val_recon) ** 2
                        val_loss += (
                            val_mse.view(val_mse.size(0), -1).sum(dim=1).mean().item()
                        )
                    count += 1
            val_loss /= count
            logger.info(f"[Iter {iteration}] Val Loss: {val_loss:.4f}")
            logger_data.add("val_loss", val_loss)

            # Save checkpoints
            checkpoint_data = {
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "loss_average": loss_average,
                "logger_data": logger_data.data,
                "scheduler_state_dict": scheduler.state_dict(),
            }

            torch.save(checkpoint_data, model_dir / "last.pth")
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(checkpoint_data, model_dir / "best.pth")
                logger.info(f"New best model saved at iteration {iteration}")

            loss_average = 0.0
            model.train()
        else:
            logger_data.add("val_loss", "")

        logger_data.write(reset=True)
