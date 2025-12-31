# Numpy Autoencoder

This repository contains a simple autoencoder implementation using only NumPy (PyTorch is used solely for loading the MNIST dataset). The autoencoder encodes and decodes handwritten digit images in the [MNIST](https://en.wikipedia.org/wiki/MNIST_database)
dataset downloaded from [torchvision][(https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html).

## Setup

I recommended creating a virtual environment with:

```shell
python -m venv venv
source venv/bin/activate
```

Then install dependencies with:

```shell
`pip install -r requirements.txt`
```

Once that's done, you can open up `example.ipynb` in VSCode (you could also use other GUIs like `jupyter notebook` but I have the project setup the way I used it in VSCode).

•	Click the kernel picker at the top right of the notebook (it might say “Python 3” by default).

•	Look for your virtual environment (it should appear as something like Python 3.x (venv)).

•	If you don’t see it, click “Select Another Kernel” → “Existing Environment” → pick your venv Python executable.

## Features

•	Pure NumPy implementation of an autoencoder

•	Minimal dependencies (PyTorch only for data loading)

•	Easy to modify and experiment with network architecture
