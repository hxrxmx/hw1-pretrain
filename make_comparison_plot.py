import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


results_dir = Path(os.getenv('RESULTS_DIR', Path(__file__).parent / 'results'))
runs = [
    ('run4_masked_lr3e4', 3e-4),
    ('run5_masked_lr4e4', 4e-4),
    ('run6_masked_lr5e4', 5e-4),
]

plt.figure(figsize=(8, 5))
losses = []

for folder, _ in runs:
    result_path = results_dir / folder / 'eval_results.json'
    with result_path.open(encoding='utf-8') as file:
        losses.append(json.load(file)['eval_loss'])

x = [learning_rate * 1e4 for _, learning_rate in runs]
plt.plot(x, losses, marker='o', linewidth=2)
for learning_rate, loss in zip(x, losses):
    plt.annotate(f'{loss:.4f}', (learning_rate, loss), xytext=(0, 9),
                 textcoords='offset points', ha='center')

plt.xlabel('learning rate, ×10⁻⁴')
plt.ylabel('eval loss')
plt.title('Learning-rate sweep, 15 minutes per run')
plt.grid(alpha=0.25)
plt.xticks(x)
plt.ylim(min(losses) - 0.025, max(losses) + 0.035)
plt.tight_layout()
plt.savefig(results_dir / 'eval_loss_comparison.png', dpi=180)
