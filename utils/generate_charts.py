"""
Generate performance visualization charts from benchmark data.
"""

import pandas as pd
import matplotlib.pyplot as plt

def plot_matmul_performance():
    """Plot matrix multiplication performance chart."""
    # Load data
    df = pd.read_csv('results/benchmarks/m4_matmul_results.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: CPU vs GPU time
    ax1.plot(df['size'], df['cpu_time_s'], marker='o', label='CPU', linewidth=2)
    ax1.plot(df['size'], df['gpu_time_s'], marker='s', label='GPU (MPS)', linewidth=2)
    ax1.set_xlabel('Matrix Size', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Matrix Multiplication: CPU vs GPU Performance', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Right plot: Speedup
    ax2.plot(df['size'], df['speedup'], marker='D', color='green', linewidth=2)
    ax2.axhline(y=1.0, color='red', linestyle='--', label='Break-even', alpha=0.7)
    ax2.fill_between(df['size'], 0, df['speedup'], where=(df['speedup'] > 1.8), 
                      alpha=0.3, color='green', label='Sweet Spot')
    ax2.set_xlabel('Matrix Size', fontsize=12)
    ax2.set_ylabel('Speedup (CPU/GPU)', fontsize=12)
    ax2.set_title('GPU Speedup Factor', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/charts/matmul_performance.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/charts/matmul_performance.png")
    plt.close()


def plot_operation_chain_performance():
    """Plot operation chain performance chart."""
    df = pd.read_csv('results/benchmarks/m4_operation_chain_results.csv')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(df))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], df['cpu_time_s'], width, 
                   label='CPU', color='steelblue', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], df['gpu_time_s'], width,
                   label='GPU (MPS)', color='coral', alpha=0.8)
    
    ax.set_xlabel('Matrix Size', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Operation Chain Performance: normalize→square→sum', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['size'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add speedup annotations
    for i, (cpu_t, gpu_t) in enumerate(zip(df['cpu_time_s'], df['gpu_time_s'])):
        speedup = cpu_t / gpu_t
        color = 'green' if speedup > 1.5 else 'red'
        ax.text(i, max(cpu_t, gpu_t) * 1.05, f'{speedup:.2f}x', 
               ha='center', fontweight='bold', color=color)
    
    plt.tight_layout()
    plt.savefig('results/charts/operation_chain_performance.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/charts/operation_chain_performance.png")
    plt.close()


def plot_performance_regimes():
    """Plot the three performance regimes."""
    df = pd.read_csv('results/benchmarks/m4_matmul_results.csv')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot speedup
    ax.plot(df['size'], df['speedup'], marker='o', linewidth=3, 
            color='darkblue', markersize=8)
    
    # Highlight regimes
    ax.axhspan(0, 1, alpha=0.2, color='red', label='Overhead Regime (GPU slower)')
    ax.axhspan(1.8, 2.5, alpha=0.2, color='green', label='Sweet Spot (GPU optimal)')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    # Annotations
    ax.annotate('Overhead\nDominates', xy=(1000, 0.76), xytext=(1500, 0.5),
                fontsize=11, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax.annotate('Sweet Spot\n2.17x faster!', xy=(5000, 2.17), xytext=(6000, 2.4),
                fontsize=11, ha='center', fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax.annotate('Memory\nBottleneck', xy=(20000, 1.06), xytext=(15000, 1.3),
                fontsize=11, ha='center', fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax.set_xlabel('Matrix Size', fontsize=13)
    ax.set_ylabel('GPU Speedup (×)', fontsize=13)
    ax.set_title('Performance Regimes: Three Distinct Zones', 
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)
    
    plt.tight_layout()
    plt.savefig('results/charts/performance_regimes.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/charts/performance_regimes.png")
    plt.close()


if __name__ == '__main__':
    print("Generating performance charts...")
    plot_matmul_performance()
    plot_operation_chain_performance()
    plot_performance_regimes()
    print("\n✅ All charts generated successfully!")
