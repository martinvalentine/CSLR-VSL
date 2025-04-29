import re
import matplotlib.pyplot as plt

# Path to your log file
log_path = "/home/martinvalentine/Desktop/CSLR-VSL/outputs/logs/baseline_res18/dev.txt"

# Lists to store epoch numbers and WER values
epochs = []
accuracies = []

# Read and parse the file
with open(log_path, 'r') as f:
    for line in f:
        match = re.search(r"Epoch (\d+), dev\s+([\d.]+)%", line)
        if match:
            epoch = int(match.group(1))
            acc = float(match.group(2))
            epochs.append(epoch)
            accuracies.append(acc)

# Find best WER points
best_index = accuracies.index(min(accuracies))

# Plotting
plt.figure(figsize=(14, 8))
plt.plot(epochs, accuracies, marker='o', linestyle='-', color='navy', label='Dev WER')

# Highlight the best WER point
plt.scatter(epochs[best_index], accuracies[best_index], color='red', s=100, label=f'Best WER: {accuracies[best_index]:.2f}%')

# (Optional) Annotate a few key points: first epoch, best, last
important_points = [0, best_index, len(epochs) - 1]
for idx in important_points:
    plt.annotate(
        f"Epoch {epochs[idx]}\n{accuracies[idx]:.2f}%",
        xy=(epochs[idx], accuracies[idx]),
        xytext=(epochs[idx], accuracies[idx] + 2),
        textcoords='data',
        arrowprops=dict(arrowstyle="->", color='gray'),
        ha='center',
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.8)
    )

plt.title("Development WER Over Epochs\n(Full Training Progress)", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("WER (%)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
