import pandas as pd
import matplotlib.pyplot as plt
from sympy import bottom_up

# Load the dataset
file_path = '/Users/karimabbas/Desktop/step_metrics_0.25.csv'
data = pd.read_csv(file_path)

# Calculate rolling means to smooth the data
rolling_window_size = 10  # Defines the smoothing window
data['train_loss_rolling_mean'] = data['train_loss'].rolling(window=rolling_window_size).mean()
data['train_accuracy_rolling_mean'] = data['train_accuracy'].rolling(window=rolling_window_size).mean()

# For validation loss, filter out rows where validation metrics are NaN (not available)
valid_metrics = data.dropna(subset=['valid_loss'])
# Calculate rolling mean for validation loss too
valid_metrics['valid_loss_rolling_mean'] = valid_metrics['valid_loss'].rolling(window=2).mean()

# Plotting the smoothed training and validation loss
plt.figure(figsize=(10, 6))

# Plot smoothed training loss
plt.plot(data['step'], data['train_loss_rolling_mean'], label='Train Loss (Smoothed)', color='red')

# Plot smoothed validation loss
plt.plot(valid_metrics['step'], valid_metrics['valid_loss_rolling_mean'], label='Validation Loss (Smoothed)', color='green')

plt.xlim(left=0)
plt.ylim(bottom=0)
plt.title('Training vs. Validation Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.show()
