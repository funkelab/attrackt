import torch

from attrackt.autoencoder.autoencoder_model import AutoencoderModel
from attrackt.autoencoder.autoencoder_model3d import AutoencoderModel3d
from attrackt.autoencoder.zarr_csv_dataset_autoencoder import ZarrCsvDatasetAutoencoder

torch.backends.cudnn.benchmark = True


def infer_autoencoder(
    dataset: ZarrCsvDatasetAutoencoder,
    model: AutoencoderModel | AutoencoderModel3d,
    model_checkpoint: str,
    device: str,
    output_csv_file_name: str,
):
    # specify dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, drop_last=False, shuffle=False
    )

    # set device
    device = torch.device(device)

    # load model
    state = torch.load(model_checkpoint, weights_only=False, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()

    predictions = []

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            im, sequence, id_, t = data
            im = im.to(device)

            embeddings = model(im, only_encode=True)
            embedding = embeddings.squeeze().cpu().numpy().flatten()

            # Convert sequence to native Python str
            sequence_value = sequence[0]
            if isinstance(sequence_value, torch.Tensor):
                sequence_value = sequence_value.item()
            sequence_str = str(sequence_value)

            id_ = int(id_[0])
            t = int(t[0])

            predictions.append([sequence_str, id_, t, *embedding])
    emb_dim = len(predictions[0]) - 3
    header = ["# sequence", "id", "t"] + [f"emb_{i}" for i in range(emb_dim)]
    with open(output_csv_file_name, "w") as f:
        f.write(" ".join(header) + "\n")
        for row in predictions:
            sequence, id_, t, *embedding = row
            f.write(
                f"{sequence} {id_} {t} "
                + " ".join(f"{v:.3f}" for v in embedding)
                + "\n"
            )
