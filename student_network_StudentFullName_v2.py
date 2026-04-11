#!/usr/bin/env python3
"""
student_network_StudentFullName_v2.py

Implements two behaviors depending on the last digit of a provided login username:
- even last digit: single-layer feed-forward perceptron demonstration (threshold activation)
- odd last digit: multilayer feed-forward network demonstration (tanh + backprop)

The script displays a pandas table of weight updates and draws a perceptron diagram
with matplotlib (no external image files created).

Usage: run and input a login username when prompted. Example usernames: user2, student3

This file follows the original structure from neural_network.py but uses unique
names including StudentFullName in class/function identifiers.
"""

from __future__ import annotations
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _draw_single_layer_perceptron_diagram(input_count: int = 4, output_count: int = 1, title: str = "Single-layer Perceptron"):
    """Draws a simple single-layer perceptron diagram using matplotlib.
    No image files are written; the plot is displayed.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title)
    ax.axis('off')

    # Positions for input nodes (left) and output nodes (right)
    left_x = 0.1
    right_x = 0.8
    y_inputs = np.linspace(0.9, 0.1, input_count)
    y_outputs = np.linspace(0.9, 0.1, output_count)

    # Draw input nodes as squares
    for i, y in enumerate(y_inputs):
        ax.add_patch(plt.Rectangle((left_x - 0.03, y - 0.03), 0.06, 0.06, fill=False))
        ax.text(left_x - 0.06, y, f"x{i+1}", va='center', ha='right')

    # Draw output nodes as circles and arrows
    for j, y_o in enumerate(y_outputs):
        circle = plt.Circle((right_x, y_o), 0.04, fill=False)
        ax.add_patch(circle)
        ax.text(right_x + 0.06, y_o, f"y{j+1}", va='center')

        # Connect every input to this output
        for y in y_inputs:
            ax.annotate('', xy=(right_x - 0.03, y_o), xytext=(left_x + 0.03, y),
                        arrowprops=dict(arrowstyle='->', linewidth=1))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()


class StudentFullNameSingleLayerPerceptron:
    """Single-layer perceptron using threshold activation and perceptron learning rule.
    Naming intentionally includes StudentFullName to meet naming requirements.
    """
    def __init__(self, input_size: int, learning_rate: float = 0.1):
        np.random.seed(1)
        self.learning_rate = learning_rate
        # initialize original synaptic weights (student's own original data)
        # provide four baseline values so we can consistently draw 4-input diagrams
        base_weights = [0.2, -0.4, 0.1, 0.05]
        self.weights = np.array(base_weights[:input_size], dtype=float)
        if self.weights.shape[0] < input_size:
            # pad zeros if needed
            self.weights = np.pad(self.weights, (0, input_size - self.weights.shape[0]))
        self.bias = 0.0

    @staticmethod
    def threshold_activation(value: float) -> int:
        return 1 if value >= 0 else 0

    def predict(self, input_vector: np.ndarray) -> int:
        net = float(np.dot(self.weights, input_vector) + self.bias)
        return self.threshold_activation(net)

    def train_and_record(self, training_inputs: np.ndarray, training_targets: np.ndarray, iterations: int = 3):
        """Perceptron learning rule; records weight/bias after each iteration into a pandas DataFrame."""
        records = []
        for it in range(1, iterations + 1):
            for x, t in zip(training_inputs, training_targets):
                out = self.predict(x)
                error = t - out
                self.weights = self.weights + self.learning_rate * error * x
                self.bias = self.bias + self.learning_rate * error

            rec = {f'w{i}': float(w) for i, w in enumerate(self.weights)}
            rec['bias'] = float(self.bias)
            rec['iteration'] = it
            records.append(rec)

        df = pd.DataFrame(records).set_index('iteration')
        return df


class StudentFullNameMultilayerNetwork:
    """Very small multilayer feed-forward network (one hidden layer) using tanh activation.
    Naming includes StudentFullName to satisfy uniqueness requirement.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        np.random.seed(1)
        self.W1 = 2 * np.random.random((input_dim, hidden_dim)) - 1
        self.W2 = 2 * np.random.random((hidden_dim, output_dim)) - 1

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1.0 - np.tanh(x) ** 2

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = self.tanh(np.dot(X, self.W1))
        self.z2 = self.tanh(np.dot(self.z1, self.W2))
        return self.z2

    def train(self, X: np.ndarray, Y: np.ndarray, iterations: int = 2000, lr: float = 0.01):
        for i in range(iterations):
            out = self.forward(X)
            # backprop
            delta2 = (Y - out) * self.tanh_derivative(out)
            delta1 = delta2.dot(self.W2.T) * self.tanh_derivative(self.z1)
            self.W2 += self.z1.T.dot(delta2) * lr
            self.W1 += X.T.dot(delta1) * lr


def _last_digit_of_username(username: str) -> int | None:
    # find last digit in the username
    for ch in reversed(username):
        if ch.isdigit():
            return int(ch)
    return None


def main():
    print("Enter your login username (letters first, then digits), e.g. epo0091 or user2:")
    username = input().strip()
    if not re.match(r'^[A-Za-z]+[0-9]+$', username):
        print("Invalid username format. Expected letters followed by digits (e.g., user2). Exiting.")
        sys.exit(1)

    last_digit = _last_digit_of_username(username)
    if last_digit is None:
        print("No digit found in username. Exiting.")
        sys.exit(1)

    print(f"Detected username: {username}, last digit: {last_digit} ( {'even' if last_digit%2==0 else 'odd'} )")

    # Even: single-layer perceptron
    if last_digit % 2 == 0:
        print("\nRunning single-layer perceptron demonstration...\n")
        # Use a consistent 4-element input vector and model so diagrams look identical
        input_vector = np.array([1.0, -1.0, 0.5, 0.0])
        model = StudentFullNameSingleLayerPerceptron(input_size=4, learning_rate=0.2)
        print("Original input vector:", input_vector)
        print("Initial weights:", model.weights)
        print("Initial bias:", model.bias)

        net_value = float(np.dot(model.weights, input_vector) + model.bias)
        activation = model.threshold_activation(net_value)
        print(f"Net value = {net_value:.4f}, Threshold activation -> output = {activation}\n")

        # Prepare an expanded original training set (4 features) so diagram and dataset match
        training_inputs = np.array([
            [1.0, -1.0, 0.5, 0.0],
            [0.0, 1.0, -0.5, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [1.0, -1.0, -0.5, 1.0]
        ])
        training_targets = np.array([1, 0, 1, 0, 1])

        df_weights = model.train_and_record(training_inputs, training_targets, iterations=3)
        print("Weight updates over iterations (rows = iteration):\n")
        print(df_weights)

        # Show the weight table using pandas (already printed); also save to variable
        # Draw the single-layer perceptron diagram (from screenshot Single-layer-perceptron.png)
        _draw_single_layer_perceptron_diagram(input_count=4, output_count=1,
                              title="Single-layer Perceptron Diagram (4 inputs)")

    else:
        print("\nRunning multilayer feed-forward network demonstration...\n")
        # Define original architecture with 4 inputs so diagram matches single-layer view
        input_dim = 4
        hidden_dim = 5
        output_dim = 1
        # Create an expanded dataset: all 4-bit combinations (16 samples)
        X = np.array([[int(b) for b in format(i, '04b')] for i in range(16)])
        # Use a non-linear target (e.g., XOR of first two bits) so hidden layer is useful
        Y = np.array([(x[0] ^ x[1]) for x in X]).reshape(-1, 1)

        net = StudentFullNameMultilayerNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        print("Initial W1 shape", net.W1.shape, "W2 shape", net.W2.shape)

        print("Training multilayer network for 5000 iterations on expanded dataset...")
        net.train(X, Y, iterations=5000, lr=0.01)

        out = net.forward(X)
        print("Final outputs (after training):\n", np.round(out, 3))

        # Draw a perceptron diagram from the screenshot multilayer-feedforward-network.png (single-layer perception view)
        # Draw the same 4-input perceptron diagram so visuals match the single-layer view
        _draw_single_layer_perceptron_diagram(input_count=4, output_count=1,
                              title="Diagram (consistent 4-input view)")


if __name__ == '__main__':
    main()
