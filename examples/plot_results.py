import argparse
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('csv', type=str, help='Path to log CSV file')
parser.add_argument('--out', type=str, default='metrics.png')
args = parser.parse_args()

log = pd.read_csv(args.csv)
fig, ax1 = plt.subplots()
ax1.plot(log['epoch'], log['time'], label='time')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Time (s)')
ax2 = ax1.twinx()
ax2.plot(log['epoch'], log['acc'], color='orange', label='acc')
ax2.set_ylabel('Accuracy')
fig.tight_layout()
plt.savefig(args.out)
