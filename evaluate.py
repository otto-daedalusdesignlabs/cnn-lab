"""Test saved CNN weights, inspect a test example, or predict an image."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import ROOT, SmallCNN, choose_device, run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Path to best.pt; defaults to the most recently saved checkpoint")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sample", type=int, help="Predict one official test image by its zero-based index")
    mode.add_argument("--image", type=Path, help="Predict a local image file")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.checkpoint is None:
        checkpoints = list((ROOT / "runs").glob("*/best.pt"))
        if not checkpoints:
            parser.error("No saved model found. Run `uv run train.py` first, or pass --checkpoint PATH.")
        args.checkpoint = max(checkpoints, key=lambda path: path.stat().st_mtime_ns)
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    torch.set_num_threads(min(8, torch.get_num_threads()))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    device = choose_device(args.device)
    channels = checkpoint["channels"]
    size = checkpoint["image_size"]
    mean = checkpoint["normalization_mean"]
    std = checkpoint["normalization_std"]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    model = SmallCNN(channels, size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Disable training-time dropout; weights stay unchanged.
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Dataset: {checkpoint['dataset']} | Device: {device}", flush=True)

    if args.image is not None:
        if not args.image.is_file():
            parser.error(f"Image does not exist: {args.image}")
        try:
            with Image.open(args.image) as original:
                image = ImageOps.exif_transpose(original).convert("L" if channels == 1 else "RGB")
                image = image.resize((size, size), Image.Resampling.BILINEAR)
                inputs = transform(image)
        except (OSError, ValueError) as error:
            parser.error(f"Cannot read image: {error}")
        print(f"Image: {args.image} (converted to {size}x{size}, {channels} channel(s))")
        print("Use images resembling the training dataset; model scores are not calibrated confidence.")
    else:
        dataset_type = {
            "fashion-mnist": datasets.FashionMNIST,
            "mnist": datasets.MNIST,
            "cifar10": datasets.CIFAR10,
        }[checkpoint["dataset"]]
        if checkpoint["dataset"] == "fashion-mnist":
            datasets.FashionMNIST.mirrors = [
                "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
                *datasets.FashionMNIST.mirrors,
            ]
        test = dataset_type(ROOT / "data", train=False, download=True, transform=transform)
        if args.sample is None:
            metrics = run_epoch(model, DataLoader(test, batch_size=args.batch_size), nn.CrossEntropyLoss(), device)
            print(f"Test images: {len(test):,} | Loss: {metrics['loss']:.4f} | Accuracy: {metrics['accuracy']:.2%}")
            return
        if not 0 <= args.sample < len(test):
            parser.error(f"sample must be between 0 and {len(test) - 1}")
        inputs, label = test[args.sample]
        actual = checkpoint["classes"][label]
        print(f"Test image #{args.sample} | Actual label: {actual}")
        # Save a viewable, unnormalized copy beside the checkpoint.
        preview = inputs * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
        preview_path = args.checkpoint.parent / f"test-sample-{args.sample}.png"
        transforms.ToPILImage()(preview.clamp(0, 1)).save(preview_path)
        print(f"Sample image: {preview_path}")

    with torch.inference_mode():
        scores = model(inputs.unsqueeze(0).to(device)).softmax(dim=1)[0].cpu()
    values, indices = scores.topk(3)
    predicted = checkpoint["classes"][indices[0].item()]
    print(f"Prediction: {predicted}")
    if args.sample is not None:
        print(f"Result: {'correct' if predicted == actual else 'incorrect'}")
    print("Top 3 model scores:")
    for value, index in zip(values.tolist(), indices.tolist()):
        print(f"  {checkpoint['classes'][index]}: {value:.1%}")


if __name__ == "__main__":
    main()
