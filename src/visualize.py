import pandas as pd
import matplotlib.pyplot as plt

# Load results
base = pd.read_csv('results/base_model_results.csv', index_col=0)
ens = pd.read_csv('results/ensemble_results.csv', index_col=0)
stk = pd.read_csv('results/stacking_results.csv', index_col=0)

# Combine
all_results = pd.concat([base, ens, stk])

# Bar plot
ax = all_results.plot.bar(rot=45, figsize=(10,6))
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
plt.tight_layout()
plt.savefig('results/model_performance.png')
print('Visualization saved to results/model_performance.png')
