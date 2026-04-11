"""Multilayer Feed-Forward Network (custom example)

This script implements a small multilayer feed-forward neural network (MLP)
with two hidden layers and a single output neuron. It uses sigmoid activations
and trains with basic batch gradient descent (backpropagation).

The network architecture and training progress are printed, and a graph of
the network architecture is saved as `network_architecture.png`.

Run with:
  conda activate base && python3 multilayer_feedforward_network_custom.py
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    return a * (1.0 - a)


class MultilayerFeedForwardNetwork:
    """A simple MLP with customizable hidden layers (two hidden layers here).

    Uses unique variable names and demonstrates training on an original dataset.
    """

    def __init__(self, input_size: int, hidden_layer_sizes: List[int], output_size: int, learning_rate_eta: float = 0.5, random_seed: int = 42):
        np.random.seed(random_seed)
        # Layer sizes: [input_size] + hidden_layer_sizes + [output_size]
        self.layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        self.num_layers = len(self.layer_sizes)
        # Initialize weight matrices and bias vectors for each connection
        # W_matrices[i] maps from layer i to layer i+1
        self.W_matrices = [np.random.uniform(-0.5, 0.5, (self.layer_sizes[i+1], self.layer_sizes[i]))
                           for i in range(self.num_layers - 1)]
        self.b_vectors = [np.random.uniform(-0.5, 0.5, (self.layer_sizes[i+1], 1))
                          for i in range(self.num_layers - 1)]
        self.learning_rate_eta = learning_rate_eta

    def forward_pass(self, input_column: np.ndarray):
        """Return list of activations (including input) and net inputs per layer."""
        activations = [input_column]
        net_inputs = []
        for W, b in zip(self.W_matrices, self.b_vectors):
            net = np.dot(W, activations[-1]) + b
            net_inputs.append(net)
            activation = sigmoid(net)
            activations.append(activation)
        return activations, net_inputs

    def predict(self, input_vector: List[float]) -> float:
        x_col = np.array(input_vector, dtype=float).reshape(-1, 1)
        activations, _ = self.forward_pass(x_col)
        return float(activations[-1].item())

    def train(self, input_matrix: np.ndarray, target_matrix: np.ndarray, epochs_n: int = 1000, verbose: bool = True):
        # Batch gradient descent (compute gradients for whole dataset each epoch)
        num_samples = input_matrix.shape[1]
        training_log = []
        for epoch in range(1, epochs_n + 1):
            # Initialize accumulators
            grad_W_accum = [np.zeros_like(W) for W in self.W_matrices]
            grad_b_accum = [np.zeros_like(b) for b in self.b_vectors]
            epoch_loss = 0.0
            for i in range(num_samples):
                x_col = input_matrix[:, i].reshape(-1, 1)
                y_col = target_matrix[:, i].reshape(-1, 1)
                activations, net_inputs = self.forward_pass(x_col)
                output_activation = activations[-1]
                # Compute simple squared error loss
                sample_loss = 0.5 * np.sum((y_col - output_activation) ** 2)
                epoch_loss += sample_loss
                # Backpropagation: compute deltas
                deltas = [None] * (self.num_layers - 1)
                # Output layer delta
                deltas[-1] = (output_activation - y_col) * sigmoid_derivative(output_activation)
                # Hidden layers (backwards)
                for l in range(len(deltas) - 2, -1, -1):
                    deltas[l] = np.dot(self.W_matrices[l+1].T, deltas[l+1]) * sigmoid_derivative(activations[l+1])
                # Accumulate gradients
                for l in range(len(self.W_matrices)):
                    grad_W_accum[l] += np.dot(deltas[l], activations[l].T)
                    grad_b_accum[l] += deltas[l]
            # Update weights and biases using average gradients
            for l in range(len(self.W_matrices)):
                self.W_matrices[l] -= (self.learning_rate_eta / num_samples) * grad_W_accum[l]
                self.b_vectors[l] -= (self.learning_rate_eta / num_samples) * grad_b_accum[l]
            avg_loss = epoch_loss / num_samples
            training_log.append((epoch, avg_loss))
            if verbose and (epoch % max(1, epochs_n // 10) == 0 or epoch == 1 or epoch == epochs_n):
                print(f"Epoch {epoch:4d}/{epochs_n}  Avg Loss = {avg_loss:.6f}")
        return training_log


def draw_network_architecture(layer_sizes: List[int], save_path: str = "network_architecture.png"):
    """Draw a simple left-to-right network diagram using matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    # compute positions
    n_layers = len(layer_sizes)
    x_spacing = 1.0 / float(n_layers - 1)
    node_positions = []
    for i, size in enumerate(layer_sizes):
        x = i * x_spacing
        y_spacing = 1.0 / float(size + 1)
        positions = []
        for j in range(size):
            y = 1.0 - (j + 1) * y_spacing
            positions.append((x, y))
            circle = plt.Circle((x, y), 0.03, fill=True, color='C0')
            ax.add_patch(circle)
        node_positions.append(positions)
        # label layer
        layer_label = "Input" if i == 0 else ("Output" if i == n_layers - 1 else f"Hidden {i}")
        ax.text(x, 1.02, f"{layer_label}\n(size={size})", ha='center')
    # draw connections
    for i in range(n_layers - 1):
        for p_from in node_positions[i]:
            for p_to in node_positions[i+1]:
                ax.plot([p_from[0], p_to[0]], [p_from[1], p_to[1]], color='gray', linewidth=0.8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, 1.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    print("Multilayer Feed-Forward Network (custom example)")
    # --- Define original dataset (unique inputs / outputs) ---
    # We'll use 3-dimensional inputs and scalar outputs in (0,1)
    input_pattern_set = np.array([
        [0.1, 0.9, 0.2],
        [0.8, 0.1, 0.5],
        [0.4, 0.6, 0.9],
        [0.7, 0.8, 0.3],
    ]).T  # shape (3, 4) - each column is a sample

    desired_response_set = np.array([[0.0, 1.0, 1.0, 0.0]])  # shape (1,4)

    # --- Define network architecture (original choices) ---
    input_dim = 3
    hidden_layers = [4, 3]  # two hidden layers: first has 4 neurons, second has 3 neurons
    output_dim = 1
    learning_rate_eta = 0.8

    # Create network with unique variable names
    nif_net = MultilayerFeedForwardNetwork(input_size=input_dim, hidden_layer_sizes=hidden_layers, output_size=output_dim, learning_rate_eta=learning_rate_eta, random_seed=7)

    # Draw architecture graph
    architecture_save = "network_architecture.png"
    draw_network_architecture(nif_net.layer_sizes, save_path=architecture_save)
    print(f"Saved network architecture graph to {architecture_save}")

    # Train and demonstrate learning
    epochs_n = 3000
    training_log = nif_net.train(input_pattern_set, desired_response_set, epochs_n=epochs_n, verbose=True)

    # Show final predictions
    print('\nFinal predictions after training:')
    for idx in range(input_pattern_set.shape[1]):
        x_vec = input_pattern_set[:, idx]
        pred = nif_net.predict(x_vec)
        print(f" Sample {idx+1}: input={x_vec.tolist()}  predicted={pred:.4f}  target={float(desired_response_set[0, idx])}")

    # Save a small training log snapshot
    print('\nTraining log sample (first 5, last 5 epochs):')
    print(training_log[:5])
    print(training_log[-5:])


if __name__ == "__main__":
    main()
