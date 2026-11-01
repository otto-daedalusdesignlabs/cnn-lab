"""Train a small CNN; run `uv run train.py --help` for options."""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent


class SmallCNN(nn.Module):
    """Convolutions learn visual patterns; the classifier predicts 10 classes."""

    def __init__(self, channels: int, image_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (image_size // 4) ** 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, images):
        return self.layers(images)


def choose_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu or --device auto.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device auto on this Mac.")
    return torch.device(requested)


def load_data(name, seed, limit):
    channels = 3 if name == "cifar10" else 1
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * channels, (0.5,) * channels),
    ])
    dataset_type = {
        "fashion-mnist": datasets.FashionMNIST,
        "mnist": datasets.MNIST,
        "cifar10": datasets.CIFAR10,
    }[name]
    if name == "fashion-mnist":
        # Official project mirror; torchvision still checks the dataset MD5s.
        datasets.FashionMNIST.mirrors = [
            "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
            *datasets.FashionMNIST.mirrors,
        ]
    full_train = dataset_type(ROOT / "data", train=True, download=True, transform=transform)
    test = dataset_type(ROOT / "data", train=False, download=True, transform=transform)
    validation_size = len(full_train) // 10
    train, validation = random_split(
        full_train,
        [len(full_train) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    if limit:
        # random_split already shuffled indices. Keep validation and test complete.
        train = Subset(train, range(min(limit, len(train))))
    return train, validation, test, channels, full_train.classes


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    count = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.detach() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum()
            count += labels.size(0)
    return {"loss": loss_sum.item() / count, "accuracy": correct.item() / count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["fashion-mnist", "mnist", "cifar10"], default="fashion-mnist")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit-train", type=int, default=0, help="Limit training images for a quick check; 0 uses all")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.limit_train < 0:
        parser.error("epochs, batch-size, and lr must be positive; limit-train must be nonnegative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Avoid excessive CPU thread overhead on small batches.
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = choose_device(args.device)
    print(f"Dataset: {args.dataset} | Device: {device} | PyTorch: {torch.__version__}", flush=True)
    train, validation, test, channels, classes = load_data(args.dataset, args.seed, args.limit_train)
    print(f"Images: {len(train):,} train / {len(validation):,} validation / {len(test):,} test", flush=True)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(validation, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)
    image_size = 32 if args.dataset == "cifar10" else 28
    model = SmallCNN(channels, image_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run_dir = ROOT / "runs" / f"{args.dataset}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    run_dir.mkdir(parents=True)
    history = []
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        history.append({"epoch": epoch, "train": training, "validation": validation_metrics})
        print(f"Epoch {epoch}/{args.epochs} | loss {training['loss']:.4f} | "
              f"train {training['accuracy']:.1%} | validation {validation_metrics['accuracy']:.1%}", flush=True)
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            torch.save({
                "model_state_dict": model.state_dict(), "channels": channels,
                "image_size": image_size,
                "classes": classes, "dataset": args.dataset, "epoch": epoch,
                "validation_accuracy": best_accuracy, "config": vars(args),
                "normalization_mean": [0.5] * channels,
                "normalization_std": [0.5] * channels,
            }, run_dir / "best.pt")
        (run_dir / "metrics.json").write_text(json.dumps({"config": vars(args), "history": history}, indent=2) + "\n")

    # Select on validation data, then evaluate once on the untouched test set.
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, device)
    (run_dir / "metrics.json").write_text(json.dumps({
        "config": vars(args), "device": str(device), "torch_version": torch.__version__,
        "history": history, "best_epoch": checkpoint["epoch"], "test": test_metrics,
    }, indent=2) + "\n")
    print(f"Best model test accuracy: {test_metrics['accuracy']:.1%}", flush=True)
    print(f"Saved model and metrics: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
