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

# Plotting
plt.figure(figsize=(14, 8))
plt.plot(epochs, accuracies, marker='o', linestyle='-', color='royalblue', linewidth=2, markersize=6, label='Dev WER')

# Annotate every point
for epoch, acc in zip(epochs, accuracies):
    plt.annotate(
        f"{acc:.2f}",
        xy=(epoch, acc),
        xytext=(0, 8),  # Slightly above
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", edgecolor='none', facecolor='white', alpha=0.6)
    )

# Professional look settings
plt.title("Development WER Over Epochs (V2 Dataset)", fontsize=18, weight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("WER (%)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.8)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
