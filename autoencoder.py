from pathlib import Path
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader


class AutoEncoder:
    """
    Simple autoencoder implemented using NumPy.

    This class defines a fully connected autoencoder architecture with
    configurable encoder/decoder depths, activation functions, and
    training hyperparameters. Designed for clarity over performance.
    """

    Activation = Literal["relu", "sigmoid"]
    RELU = "relu"
    SIGMOID = "sigmoid"

    def __init__(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        encoder_layer_sizes: list[int],
        decoder_layer_sizes: list[int],
        latent_layer_size: int,
        activation_func: Activation,
    ) -> None:
        """
        Initialize the autoencoder architecture and training configuration.

        The model is kept simple to
        make operations more explicit. Encoder, latent, and decoder layers are
        defined separately for flexibility, then combined for simplicity of use.
        Weights, biases, and activations are initialized up front for clarity.
        The goal here is to allow easy experimentation with the architecture.

        A random seed is set for reproducibility. And pre-activations (computed
        as x @ W + b) are stored for every layer except the input layer.
        """

        np.random.seed(42)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.layer_sizes = [
            *encoder_layer_sizes,
            latent_layer_size,
            *decoder_layer_sizes,
        ]
        self.biases = [np.zeros(layer) for layer in self.layer_sizes[1:]]
        self.pre_activations = [np.empty(layer) for layer in self.layer_sizes[1:]]
        self.activations = [np.empty(layer) for layer in self.layer_sizes]
        self.activation_func = activation_func
        self.weights = self.init_weights()

    def init_weights(self) -> list[np.ndarray]:
        """
        Initialize weights for all layers in the autoencoder.

        Weights are stored as a list of NumPy arrays, one per layer transition. Initialization is activation-aware:
        - ReLU: He initialization (scaled normal)
        https://en.wikipedia.org/wiki/Weight_initialization#He_initialization
        - Sigmoid: Glorot initialization (scaled uniform)
        https://en.wikipedia.org/wiki/Weight_initialization#Glorot_initialization

        For more information, see the weight_initialization.md file.
        """

        weights = []

        rng = np.random.default_rng()

        for idx in range(len(self.layer_sizes) - 1):
            neurons_in = self.layer_sizes[idx]
            neurons_out = self.layer_sizes[idx + 1]

            # He initialization for ReLU
            if self.activation_func == self.RELU:
                scale = np.sqrt(2.0 / neurons_in)
                w = scale * rng.standard_normal((neurons_in, neurons_out))

            # Glorot initialization for Sigmoid
            elif self.activation_func == self.SIGMOID:
                bound = np.sqrt(6.0 / (neurons_in + neurons_out))
                w = rng.uniform(-bound, bound, size=(neurons_in, neurons_out))

            else:
                raise ValueError(f"Recieved an unknown activation function.")

            weights.append(w)

        return weights

    def get_mnist_loader(self, shuffle: bool, train: bool = True) -> DataLoader:
        """
        Get the DataLoader for the MNIST dataset.

        Args:
            shuffle (bool): Whether to shuffle the dataset.
            train (bool, default=True): Whether to load the training split (True) or
                test split (False).
        """

        MNIST_DIR_PATH = Path("mnist")
        MNIST_DATA_PATH = MNIST_DIR_PATH / "MNIST" / "raw"
        download = False if MNIST_DATA_PATH.exists() else True

        mnist_trans = transforms.Compose([transforms.ToTensor()])

        mnist_data = MNIST(
            str(MNIST_DIR_PATH), download=download, transform=mnist_trans, train=train
        )

        data_loader = DataLoader(
            mnist_data,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

        return data_loader

    def reconstruction_loss(self, input: np.ndarray, output: np.ndarray) -> float:
        """
        Mean Squared Error (MSE) reconstruction loss between the model output and the input.

        Args:
            input (np.ndarray): Original input data of shape [batch_size, n_features].
            output (np.ndarray): Reconstructed output of the same shape.

        Returns:
            float: The average mean squared error for the batch.
        """

        if input.shape != output.shape:
            raise ValueError(
                "Reconstruction loss requires matching shapes, "
                f"got {input.shape=}, {output.shape=}."
            )

        return np.mean((output - input) ** 2)

    def d_loss_d_out(self, input: np.ndarray, output: np.ndarray) -> np.ndarray:
        """
        Derivative of the MSE reconstruction loss with respect to the model output.

        For MSE:
            loss = 1/N sum((output - input)^2)
            d(loss)/d(output) = 2/N * (output - input)

        Args:
            input (np.ndarray): Original input data of shape [batch_size, n_features].
            output (np.ndarray): Reconstructed output of the same shape.

        Returns:
            np.ndarray: Gradient of shape [batch_size, n_features].
        """

        if input.shape != output.shape:
            raise ValueError(
                "Derivative of reconstruction loss requires matching shapes, "
                f"got {input.shape=}, {output.shape=}."
            )

        batch_size, _ = input.shape
        return 2 * (output - input) / batch_size

    def apply_activation(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the activation function to the given pre-activation array.

        This function is used during the forward pass to introduce non-linearity
        into the network and thus learn more complex patterns. Without this, the
        network would just be a linear combination of matrix multiplications which
        is just learning a linear combination of the input.

        Uses the numerically stable version of the sigmoid function, which helps
        sigmoid avoid errors due to floating point arithmetic.
        """

        if self.activation_func == self.RELU:
            return np.maximum(0, z)

        elif self.activation_func == self.SIGMOID:
            return 1 / (1 + np.exp(-z))

        else:
            raise ValueError(f"Unsupported activation function: {self.activation_func}")

    def d_activation_dz(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the activation function with respect to pre-activation z

        For ReLU: derivative is 1 for positive z, 0 otherwise.
        For Sigmoid: derivative is sigmoid(z) * (1 - sigmoid(z)).

        Used during backprop to propagate gradients through the non-linearity.
        """

        if self.activation_func == self.RELU:
            return 1.0 * (z > 0)

        elif self.activation_func == self.SIGMOID:
            sigmoid_output = self.apply_activation(z)
            return sigmoid_output * (1 - sigmoid_output)

        else:
            raise ValueError(f"Unsupported activation function: {self.activation_func}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass through the autoencoder.

        For each layer, matrix multiply the inputs and weights then add
        biases to get pre-activations. Then, apply the non-linear activation
        function. Both activations and pre-activations are stored for
        use during backpropagation.

        Args:
            x (np.ndarray): Input data of shape [batch_size, n_input].

        Returns:
            np.ndarray: Output of the autoencoder of shape [batch_size, n_output].
        """

        self.activations[0] = x

        for layer_i, (w, b) in enumerate(zip(self.weights, self.biases)):

            z = x @ w + b
            self.pre_activations[layer_i] = z
            x = self.apply_activation(z)
            self.activations[layer_i + 1] = x

        return x

    def backward(self, dl_dout: np.ndarray) -> None:
        """
        Perform back-propagation through the autoencoder to update weights and biases.

        The method computes the gradient of the reconstruction loss with respect to
        each layer's weights and biases using the chain rule. It uses the stored
        activations and pre-activations from the forward pass. Updates are applied
        using gradient descent.

        Assumes:
            - self.activations contains activations for all layers including input.
            - self.pre_activations contains the pre-activation (linear) outputs.

        For more information on the math using a similar style, see the backprop.md file.
        """

        z_out = self.pre_activations[-1]

        # out = act(z_out)
        # dl/dz_out = dl/dout * dout/dz_out
        # dl/dz_out = dl/dout * dact/dz_out
        dl_dz_values = [dl_dout * self.d_activation_dz(z_out)]

        # Backpropagate through hidden layers
        for layer_i in range(len(self.weights) - 1, 0, -1):

            z_prev = self.pre_activations[layer_i - 1]
            W = self.weights[layer_i]

            # dl/dz_{i-1} = dl/dx_{i-1} * dact/dz_{i-1}
            # dl/dx_{i-1} = dl/dz_i @ W_i.T
            # dl/dz_{i-1} = dl/dz_i @ W_i.T * dact/dz_{i-1}
            dl_dz_prev = dl_dz_values[-1].dot(W.T) * self.d_activation_dz(z_prev)

            dl_dz_values.append(dl_dz_prev)

        # Reverse to match layer order
        dl_dz_values.reverse()

        # Apply gradient updates to weights and biases
        # The -= applies gradient descent, moving the parameters in the direction
        # that reduces the loss
        for layer_i in range(len(self.weights)):

            # Note that len(activations) = len(weights) + 1 so the
            # same index lines up w_i with x_{i-1}
            # Gradient for weights: dl/dW_i = x_{i-1}.T @ dl/dz_i
            self.weights[layer_i] -= self.learning_rate * self.activations[
                layer_i
            ].T.dot(dl_dz_values[layer_i])

            # Gradient for biases: mean over batch of dl/dz_i
            self.biases[layer_i] -= self.learning_rate * np.mean(
                dl_dz_values[layer_i], axis=0
            )

    def train(self) -> None:
        """
        Train the autoencoder

        All calculations are done on the cpu for simplicity. Loss is normalized
        by batch size so it can be compared across batches.
        """

        loader = self.get_mnist_loader(shuffle=True, train=True)

        loss_values = []

        for epoch in tqdm(range(self.epochs)):

            # Track per-batch losses for this epoch
            total_epoch_loss = 0

            for images, _ in loader:

                # Perform forward and backward passes then record loss
                images = images.view(images.size(0), -1).numpy()
                out = self.forward(images)
                dl_dout = self.d_loss_d_out(images, out)
                self.backward(dl_dout)
                total_epoch_loss += self.reconstruction_loss(images, out)

            epoch_average_loss = total_epoch_loss / len(loader)
            loss_values.append(epoch_average_loss)
            print(f"Epoch {epoch+1}/{self.epochs}, loss: {epoch_average_loss:.4f}")

        print("Training complete.")

        return loss_values

    def show_loss_curve(self, losses: list[float]) -> None:
        """Plot the epoch losses using matplotlib"""

        plt.plot(range(1, len(losses) + 1), losses)
        plt.title("Loss curve after training")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def visualize(self, train=False, num_imgs=5) -> None:
        """
        Visualize results using first few samples out of train or eval data

        This function will show an image and next to it the autoencoder's
        recreation of that same image.
        """

        loader = self.get_mnist_loader(shuffle=False, train=train)

        for images, _ in loader:
            images = images.view(images.size(0), -1).numpy()
            output = self.forward(images)

            for i, (img, out) in enumerate(zip(images, output)):
                if i >= num_imgs:
                    break

                plt.figure(figsize=(4, 2))

                # Input image
                plt.subplot(1, 2, 1)
                plt.imshow(img.reshape(28, 28), cmap="gray", interpolation="nearest")
                plt.title(f"Input {i+1}")
                plt.axis("off")

                # Reconstructed output
                plt.subplot(1, 2, 2)
                plt.imshow(out.reshape(28, 28), cmap="gray", interpolation="nearest")
                plt.title(f"Output {i+1}")
                plt.axis("off")

                plt.show()

            # Only take the first batch
            break
