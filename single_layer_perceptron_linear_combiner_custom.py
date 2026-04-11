"""Single-layer perceptron / linear combiner example (custom data)

This script defines original input vectors, initial synaptic weights, and bias.
It uses the Signum activation function (returns +1 or -1) and demonstrates
the output calculation and weight updates over several iterations.

To run (use conda base environment as required):
  conda activate base && python3 single_layer_perceptron_linear_combiner_custom.py
"""
from typing import List, Tuple
import pandas as pd


def signum(x: float) -> int:
    """Signum activation: returns +1 for x >= 0, else -1."""
    return 1 if x >= 0 else -1


class SingleLayerPerceptron:
    def __init__(self, initial_weights: List[float], initial_bias: float, learning_rate: float = 0.1):
        self.weights = initial_weights[:]  # copy
        self.bias = initial_bias
        self.learning_rate = learning_rate

    def net_input(self, input_vector: List[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, input_vector)) + self.bias

    def predict(self, input_vector: List[float]) -> int:
        net = self.net_input(input_vector)
        return signum(net)

    def train_one_sample(self, input_vector: List[float], target: int) -> Tuple[List[float], float, int]:
        output = self.predict(input_vector)
        error = target - output
        # Weight update: w <- w + learning_rate * error * x
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * input_vector[i]
        # Bias update: b <- b + learning_rate * error
        self.bias += self.learning_rate * error
        return self.weights[:], self.bias, error


def format_weights_line(step: int, weights: List[float], bias: float, error: int) -> str:
    weights_str = ", ".join(f"{w:+0.4f}" for w in weights)
    return f"{step:3d} | {weights_str} | {bias:+0.4f} | {error:2d}"


def main():
    # --- Custom original data (user-defined) ---
    # Each input vector has 3 features (for example purposes).
    input_vectors = [
        [1.0, 0.0, 1.0],  # custom input #1
        [0.0, 1.0, 1.0],  # custom input #2
        [1.0, 1.0, 0.0],  # custom input #3
        [0.0, 0.0, 1.0],  # custom input #4
    ]

    # Targets using Signum output space {+1, -1}
    target_outputs = [1, -1, 1, -1]

    # Initial synaptic weights and bias (original, user-chosen)
    initial_synaptic_weights = [0.20, -0.10, 0.05]
    initial_bias_value = 0.10

    # Learning rate
    learning_rate_value = 0.1

    perceptron = SingleLayerPerceptron(
        initial_weights=initial_synaptic_weights,
        initial_bias=initial_bias_value,
        learning_rate=learning_rate_value,
    )

    print("Single-Layer Perceptron (Linear Combiner) - Custom Example")
    print("Activation function: Signum (output +1 if net >= 0 else -1)")
    print()

    # Demonstrate calculation of the output for the first sample (manual step)
    sample_index = 0
    sample = input_vectors[sample_index]
    net_first = perceptron.net_input(sample)
    output_first = perceptron.predict(sample)
    print("Demonstration (manual calc) for first input vector:")
    print(f" Input vector: {sample}")
    print(f" Initial weights: {initial_synaptic_weights}, Initial bias: {initial_bias_value:+0.4f}")
    print(f" Net input = sum(w*x) + bias = {net_first:+0.4f}")
    print(f" Activation (Signum) -> output = {output_first}")
    print()

    # Train for several iterations and show weight adjustments
    max_epochs = 3  # at least three iterations (epochs)
    print("Weight update log (each row = update step):")
    print("Step | Weights (w1, w2, w3)           | Bias   | Err")
    print("---- | ------------------------------- | ------ | ---")

    step = 0
    # Prepare rows for pandas DataFrame
    weight_update_rows = []
    # Log initial state as step 0
    print(format_weights_line(step, perceptron.weights, perceptron.bias, 0))
    weight_update_rows.append({
        "step": step,
        "w1": perceptron.weights[0],
        "w2": perceptron.weights[1],
        "w3": perceptron.weights[2],
        "bias": perceptron.bias,
        "error": 0,
        "sample_index": None,
        "epoch": 0,
        "net_before": None,
        "out_before": None,
    })

    # Run training: iterate through the dataset for max_epochs
    for epoch in range(1, max_epochs + 1):
        for idx, (x_vec, t) in enumerate(zip(input_vectors, target_outputs), start=1):
            step += 1
            weights_before = perceptron.weights[:]
            bias_before = perceptron.bias
            new_weights, new_bias, error = perceptron.train_one_sample(x_vec, t)
            # Show net and output for this sample as demonstration
            net_val = sum(w * xi for w, xi in zip(weights_before, x_vec)) + bias_before
            predicted_before = signum(net_val)
            print(format_weights_line(step, new_weights, new_bias, error) +
                f"   <-- sample {idx} epoch {epoch} | x={x_vec} t={t} net_before={net_val:+0.4f} out_before={predicted_before}")
            weight_update_rows.append({
                "step": step,
                "w1": new_weights[0],
                "w2": new_weights[1],
                "w3": new_weights[2],
                "bias": new_bias,
                "error": error,
                "sample_index": idx,
                "epoch": epoch,
                "net_before": net_val,
                "out_before": predicted_before,
            })

    print()
    print("Final learned weights and bias:")
    print(f" Weights: {perceptron.weights}")
    print(f" Bias: {perceptron.bias:+0.4f}")

    # Final predictions on the dataset
    print()
    print("Final predictions on the dataset:")
    for i, x_vec in enumerate(input_vectors, start=1):
        net_val = perceptron.net_input(x_vec)
        out_val = perceptron.predict(x_vec)
        print(f" Sample {i}: x={x_vec} net={net_val:+0.4f} output={out_val} target={target_outputs[i-1]}")

    # Create pandas DataFrame from the collected rows and display/save it
    df = pd.DataFrame(weight_update_rows)
    # Reorder columns for readability
    cols = ["step", "epoch", "sample_index", "w1", "w2", "w3", "bias", "error", "net_before", "out_before"]
    df = df[cols]
    print()
    print("Weight update table (pandas DataFrame):")
    print(df.to_string(index=False))
    # Save to CSV
    csv_path = "weight_update_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved weight update table to {csv_path}")


if __name__ == "__main__":
    main()
