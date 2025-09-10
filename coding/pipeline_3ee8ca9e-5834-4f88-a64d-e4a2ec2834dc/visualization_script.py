import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Load the data
historical_data = pd.read_csv('3ee8ca9e-5834-4f88-a64d-e4a2ec2834dc.csv')
predictions_data = pd.read_csv('predictions.csv')

# Create a comprehensive anomaly detection visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Financial Transaction Anomaly Detection Analysis', fontsize=16, fontweight='bold')

# 1. Anomaly Score Distribution
ax1 = axes[0, 0]
anomaly_scores = predictions_data['anomaly_score']
is_anomaly = predictions_data['is_anomaly']

# Create histogram with different colors for anomalies vs normal
normal_scores = anomaly_scores[is_anomaly == 0]
anomaly_scores_vals = anomaly_scores[is_anomaly == 1]

ax1.hist(normal_scores, bins=50, alpha=0.7, label='Normal Transactions', color='skyblue')
ax1.hist(anomaly_scores_vals, bins=50, alpha=0.7, label='Anomalies', color='red')
ax1.axvline(anomaly_scores.quantile(0.05), color='darkred', linestyle='--', 
           label=f'5% Threshold: {anomaly_scores.quantile(0.05):.3f}')
ax1.set_xlabel('Anomaly Score (lower = more anomalous)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Anomaly Scores')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Top Anomalous Transactions by Amount
ax2 = axes[0, 1]
top_anomalies = predictions_data.nlargest(10, 'factura_monto_total')[['factura_nombre', 'factura_monto_total', 'anomaly_score', 'is_anomaly']]
top_anomalies = top_anomalies.sort_values('factura_monto_total', ascending=True)

colors = ['red' if anomaly else 'skyblue' for anomaly in top_anomalies['is_anomaly']]
bars = ax2.barh(range(len(top_anomalies)), top_anomalies['factura_monto_total'], color=colors)
ax2.set_yticks(range(len(top_anomalies)))
ax2.set_yticklabels(top_anomalies['factura_nombre'].str[:30] + '...')
ax2.set_xlabel('Transaction Amount')
ax2.set_title('Top 10 Largest Transactions (Red = Anomaly)')
ax2.grid(True, alpha=0.3)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, top_anomalies['anomaly_score'])):
    ax2.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2, 
            f'score: {score:.3f}', va='center', fontsize=8)

# 3. Anomaly Rate by Invoice Prefix
ax3 = axes[1, 0]
if 'invoice_prefix' in predictions_data.columns:
    prefix_anomaly = predictions_data.groupby('invoice_prefix').agg({
        'is_anomaly': ['count', 'mean']
    }).round(3)
    prefix_anomaly.columns = ['count', 'anomaly_rate']
    prefix_anomaly = prefix_anomaly.nlargest(10, 'count')
    
    x = range(len(prefix_anomaly))
    ax3.bar(x, prefix_anomaly['count'], alpha=0.6, label='Total Transactions')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x, prefix_anomaly['anomaly_rate'] * 100, 'ro-', linewidth=2, markersize=6, label='Anomaly Rate %')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(prefix_anomaly.index, rotation=45, ha='right')
    ax3.set_ylabel('Transaction Count')
    ax3_twin.set_ylabel('Anomaly Rate (%)', color='red')
    ax3.set_title('Anomaly Rate by Invoice Prefix (Top 10 by Count)')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

# 4. Payment Behavior Analysis
ax4 = axes[1, 1]
payment_analysis = predictions_data.groupby('is_anomaly').agg({
    'payment_ratio': 'mean',
    'is_zero_payment': 'mean',
    'is_double_payment': 'mean'
}).round(3)

x = np.arange(len(payment_analysis))
width = 0.25

bars1 = ax4.bar(x - width, payment_analysis['payment_ratio'], width, label='Avg Payment Ratio', alpha=0.8)
bars2 = ax4.bar(x, payment_analysis['is_zero_payment'], width, label='Zero Payment %', alpha=0.8)
bars3 = ax4.bar(x + width, payment_analysis['is_double_payment'], width, label='Double Payment %', alpha=0.8)

ax4.set_xlabel('Transaction Type')
ax4.set_ylabel('Percentage/Ratio')
ax4.set_title('Payment Behavior: Normal vs Anomalous Transactions')
ax4.set_xticks(x)
ax4.set_xticklabels(['Normal', 'Anomalous'])
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('forecast_plot.png', bbox_inches='tight', dpi=300)
plt.close()

print("Anomaly detection visualization saved as forecast_plot.png")