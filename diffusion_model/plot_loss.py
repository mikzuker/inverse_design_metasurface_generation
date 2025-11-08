import matplotlib.pyplot as plt
import numpy as np

# Read data from file
with open("loss_20000_500.txt", "r") as f:
    loss_values = [float(line.strip()) for line in f if line.strip()]

max_iterations = 40000
loss_values = loss_values[:max_iterations]

# Parameters for epoch calculation
train_batch_size = 16
gradient_accumulate_every = 2
dataset_size = 11000

# Calculate epochs for each iteration
samples_per_step = train_batch_size * gradient_accumulate_every
iterations = np.arange(1, len(loss_values) + 1)
epochs = (iterations * samples_per_step) / dataset_size

# Create plot
plt.figure(figsize=(12, 6))

# Calculate trend line (moving average)
window_size = 250
if len(loss_values) >= window_size:
    trend_line = np.convolve(
        loss_values, np.ones(window_size) / window_size, mode="valid"
    )
    trend_epochs = epochs[window_size - 1 :]
else:
    trend_line = loss_values
    trend_epochs = epochs

# Main loss plot
plt.plot(epochs, loss_values, "b-", linewidth=1, alpha=0.6, label="Loss")
plt.plot(
    trend_epochs,
    trend_line,
    "r-",
    linewidth=3,
    alpha=1,
    label=f"Trend (window={window_size})",
)
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.title("Training Loss Over Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("loss_plot_epochs_20000_500.png", dpi=300, bbox_inches="tight")
plt.show()

# Print statistics
print(f"Total iterations: {len(loss_values)}")
print(f"Initial loss: {loss_values[0]:.6f}")
print(f"Final loss: {loss_values[-1]:.6f}")
print(f"Minimum loss: {min(loss_values):.6f}")
print(f"Maximum loss: {max(loss_values):.6f}")
print(f"Average loss: {np.mean(loss_values):.6f}")
print(f"Standard deviation: {np.std(loss_values):.6f}")

# Find iteration with minimum loss
min_loss_idx = np.argmin(loss_values)
min_loss_epoch = epochs[min_loss_idx]
print(
    f"Minimum loss achieved at iteration {min_loss_idx + 1} (epoch {min_loss_epoch:.0f})"
)

# Trend analysis
if len(trend_line) > 1:
    trend_start = trend_line[0]
    trend_end = trend_line[-1]
    trend_change = trend_end - trend_start
    trend_percent_change = (trend_change / trend_start) * 100 if trend_start != 0 else 0

    print(f"\n=== TREND ANALYSIS (window {window_size}) ===")
    print(f"Initial trend value: {trend_start:.6f}")
    print(f"Final trend value: {trend_end:.6f}")
    print(f"Trend change: {trend_change:.6f}")
    print(f"Trend percentage change: {trend_percent_change:.2f}%")

    if trend_change > 0:
        print("Trend: INCREASE in loss (possible overfitting)")
    elif trend_change < 0:
        print("Trend: DECREASE in loss (model improvement)")
    else:
        print("Trend: STABLE (no changes)")
