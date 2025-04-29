import matplotlib.pyplot as plt
import re

# Path to the log file
log_file_path = '/home/martinvalentine/Desktop/CSLR-VSL/outputs/logs/baseline_res18/log.txt'

# Lists to store extracted data
epochs = []
train_losses = []
eval_losses = []

# Read the log file and process it
with open(log_file_path, 'r') as file:
    lines = file.readlines()

    for line in lines:
        # Match Epoch number
        epoch_match = re.search(r"Epoch: (\d+)", line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            epochs.append(epoch)

        # Match training loss
        train_loss_match = re.search(r"Loss: ([\d\.]+)", line)
        if train_loss_match:
            train_losses.append(float(train_loss_match.group(1)))

        # Match evaluation loss and remove trailing period before converting to float
        eval_loss_match = re.search(r"Mean evaluation loss: ([\d\.]+)", line)
        if eval_loss_match:
            eval_loss_str = eval_loss_match.group(1)
            eval_loss_str = eval_loss_str.rstrip('.')  # Remove any trailing period
            eval_losses.append(float(eval_loss_str))

# Check if lengths match, otherwise truncate
min_len = min(len(epochs), len(train_losses), len(eval_losses))
epochs = epochs[:min_len]
train_losses = train_losses[:min_len]
eval_losses = eval_losses[:min_len]

# Plotting the losses
plt.figure(figsize=(10, 6))

# Plot Training Loss
train_line, = plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o')

# Plot Validation Loss
eval_line, = plt.plot(epochs, eval_losses, label='Validation Loss', color='orange', marker='o')

# Add labels and title
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Validation Loss over Epochs', fontsize=16)
plt.legend(fontsize=12)

# Annotate the values on the plot
for i, txt in enumerate(train_losses):
    plt.annotate(f'{txt:.2f}', (epochs[i], train_losses[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=10)
for i, txt in enumerate(eval_losses):
    plt.annotate(f'{txt:.2f}', (epochs[i], eval_losses[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=10)

# Add grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Display the plot
plt.tight_layout()  # Ensure everything fits well
plt.show()
