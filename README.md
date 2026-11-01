# CNN Lab: Teach a Neural Network to See

**Start with pixels. Train a model. Find out what it learned.**

Build your first image classifier with a small, readable PyTorch project. Train
it to recognize handwritten digits, clothing, or everyday objects, then change
the network and compare the results. The training code lives in one Python file
so you can follow the whole process.

Datasets download automatically. Apple Silicon GPUs, NVIDIA GPUs, and CPUs are
supported through automatic device selection.

## Your first training run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
git clone https://github.com/otto-daedalusdesignlabs/cnn-lab.git
cd cnn-lab
uv sync --locked
uv run train.py --epochs 1 --limit-train 2048
```

This quick check trains on 2,048 Fashion-MNIST images, evaluates the model, and
saves its weights and results. The first run needs an internet connection to
download dependencies and data. No dataset account or API key is needed.

Ready to use the full training split? Run:

```sh
uv run train.py --dataset fashion-mnist --epochs 5
```

An **epoch** is one pass through the training data. After each epoch, the script
prints training loss and training and validation accuracy. At the end, it reports
the best model's accuracy on a separate test set.

The project pins Python 3.13 in `.python-version`; uv can download it if needed.
The package declares Python 3.11 or newer. GPU availability depends on your
hardware, operating system, and PyTorch installation. See the
[PyTorch installation guide](https://pytorch.org/get-started/locally/) for platform
requirements.

## Test your saved model

You can evaluate existing weights without retraining. By default, these commands
load the most recently saved `best.pt` under `runs/` and print its path and dataset.
To choose a specific model, add `--checkpoint path/to/best.pt`.

```sh
# Evaluate the complete official test set
uv run evaluate.py

# Inspect a single test example, with its true label and top 3 predictions
uv run evaluate.py --sample 0

# Try your own image
uv run evaluate.py --image /path/to/image.png
```

Sample mode also saves `test-sample-0.png` beside the checkpoint so you can open
the image you tested. You can choose other examples with `--sample 1`, `--sample 2`,
and so on. Full evaluation prints accuracy and loss; these commands do not update
model weights.

Use a model trained for the image you want to test: the Fashion-MNIST model
recognizes clothing categories, while an MNIST model recognizes digits. Custom
images are converted to the model's color mode, resized, and normalized. They
should resemble the training images. Fashion-MNIST uses small grayscale product
images on dark backgrounds; ordinary phone photos may perform poorly. The top 3
percentages are model scores, not calibrated probabilities that an answer is right.

A fresh clone has no saved weights because `runs/` stays local. Train a model first
or provide a compatible checkpoint from another run.

## Choose what your model learns

| Dataset | What it recognizes | Image format | Official train / test images | Start here if… |
| --- | --- | --- | --- | --- |
| [MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) | Handwritten digits, 0–9 | 28 × 28 grayscale | 60,000 / 10,000 | You want the easiest introduction. |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) | 10 categories of clothing, shoes, and bags | 28 × 28 grayscale | 60,000 / 10,000 | You want a good first challenge. |
| [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html) | 10 categories of animals and vehicles | 32 × 32 color | 50,000 / 10,000 | You are ready for more varied images. |

```sh
# Begin with handwritten digits
uv run train.py --dataset mnist --epochs 3

# Learn to classify clothing (the default dataset)
uv run train.py --dataset fashion-mnist --epochs 5

# Try color images of animals and vehicles
uv run train.py --dataset cifar10 --epochs 15
```

The script reserves 10% of each official training set for validation. The
`--limit-train` option reduces only the training split; validation and test data
remain complete. Downloads are cached in `data/` and reused on later runs.
Fashion-MNIST uses its official GitHub project mirror with torchvision integrity
checks. Dataset licenses and terms are available from the linked sources.

## What is a CNN?

A **convolutional neural network** learns patterns from images. Its convolution
layers apply small filters across the image, learning useful features during
training. Pooling reduces the size of those feature maps, and a classifier uses
the resulting features to predict one of the ten categories.

This project's network follows a simple path:

```text
Image → Convolution + ReLU → Max pooling
      → Convolution + ReLU → Max pooling
      → Flatten → Dense layer + ReLU → Dropout → 10 class scores
```

The network uses 32 filters in the first convolution and 64 in the second.
Inputs are normalized, and training uses the Adam optimizer and cross-entropy
loss. You can read and edit the network in [`train.py`](train.py).

## Read your results

Every run gets its own folder:

```text
runs/
└── fashion-mnist-<timestamp>/
    ├── best.pt         # Best model selected using validation accuracy
    └── metrics.json    # Settings, epoch history, and final test results
```

- **Loss** measures prediction error. Lower is generally better.
- **Training accuracy** measures performance on images used to update the model.
- **Validation accuracy** measures performance on held-out images and selects
  the checkpoint to keep.
- **Test accuracy** evaluates that selected checkpoint once on the official
  test set.

The checkpoint contains model weights, class names, input dimensions,
normalization settings, and run configuration. Use `evaluate.py` to reload it for
evaluation or individual predictions.

### A real first run

An initial check on an Apple M5 Pro with 48 GiB of unified memory used the Metal
GPU (`mps`), PyTorch 2.15.0, seed 42, and the quick-start command above:

| Measurement | Result |
| --- | --- |
| Training images | 2,048 |
| Epochs | 1 |
| Validation accuracy | 65.32% |
| Test accuracy | 66.31% |

The recorded metrics are in
[`examples/fashion-mnist-smoke.json`](examples/fashion-mnist-smoke.json).
This small run verifies the training workflow; it is not a full-dataset
benchmark or an accuracy promise. Results can vary across devices and library
versions, even with the same seed.

## Make it your experiment

Try one change at a time so you can tell what helped:

1. Train longer and watch whether validation accuracy keeps improving.
2. Change the learning rate with `--lr 0.0003`.
3. Compare batch sizes with `--batch-size 64` or `--batch-size 256`.
4. Edit the convolution widths or dropout rate in `SmallCNN`.
5. Move from grayscale images to CIFAR-10.

Use validation results to compare experiments. Keep the test set for final
evaluation. CIFAR-10 is a harder task; stronger results usually need additional
training, data augmentation, and model changes.

### Training options

| Option | Default | Purpose |
| --- | --- | --- |
| `--dataset` | `fashion-mnist` | Choose `mnist`, `fashion-mnist`, or `cifar10`. |
| `--epochs` | `5` | Number of passes through the training data. |
| `--batch-size` | `128` | Images processed in each training step. |
| `--lr` | `0.001` | Optimizer learning rate. |
| `--seed` | `42` | Seed for model initialization, splits, and shuffling. |
| `--device` | `auto` | Choose `auto`, `mps`, `cuda`, or `cpu`. |
| `--limit-train` | `0` | Cap training images for a quick check; `0` uses all. |

```sh
uv run train.py --help
```

Automatic device selection prefers CUDA, then Apple MPS, then CPU. To force CPU
execution, add `--device cpu`. For Apple GPU setup, see
[PyTorch's MPS guide](https://docs.pytorch.org/docs/stable/notes/mps.html).

## Find your way around

| File | Purpose |
| --- | --- |
| [`train.py`](train.py) | Network, dataset loading, training, and evaluation. |
| [`evaluate.py`](evaluate.py) | Test saved weights or predict a single image. |
| [`pyproject.toml`](pyproject.toml) | Python requirements and dependencies. |
| [`uv.lock`](uv.lock) | Locked dependency versions for repeatable setup. |
| [`.python-version`](.python-version) | Default Python version for uv. |
| [`examples/fashion-mnist-smoke.json`](examples/fashion-mnist-smoke.json) | Recorded results from the initial training check. |

Downloaded datasets, generated checkpoints, run folders, and the local Python
environment stay outside version control. They are created locally when you run
the project.
