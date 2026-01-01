# Numpy Autoencoder

This repository contains a simple autoencoder implementation using only NumPy (PyTorch is used solely for loading the MNIST dataset). The autoencoder encodes and decodes handwritten digit images in the [MNIST](https://en.wikipedia.org/wiki/MNIST_database)
dataset downloaded from [torchvision](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html).

## Setup

This project uses [pyenv](https://github.com/pyenv/pyenv) for Python version management. From any directory run:

```shell
pyenv install 3.11.8
```

Within the project directory create a virtual environment with:

```shell
python -m venv venv
source venv/bin/activate
```

Then install dependencies with:

```shell
pip install -r requirements.txt
```

Once that's done, you can open up `example.ipynb` in VSCode (you could also use other GUIs like `jupyter notebook` but I have the project setup with the packages I needed to run it in VSCode).

• Install the Jypter extension if you don't already have it.

• Click the kernel picker at the top right of the notebook and select Python Environment.

• Select your virtual environment (it should appear as something like venv (3.10.12)).

## Features

• Pure NumPy implementation of an autoencoder

• Minimal dependencies (PyTorch only for data loading)

• Easy to modify and experiment with network architecture
