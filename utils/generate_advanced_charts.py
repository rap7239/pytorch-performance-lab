"""
Generate advanced performance visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_throughput_scaling():
    """Plot throughput (operations per second) scaling."""
    df = pd.read_csv('results/benchmarks/m4_matmul_results.csv')
    
    # Calculate throughput (GFLOPS)
    # Matrix multiply: 2*n^3 operations
    df['cpu_gflops'] = (2 * df['size']**3) / (df['cpu_time_s'] * 1e9)
    df['gpu_gflops'] = (2 * df['size']**3) / (df['gpu_time_s'] * 1e9)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['size'], df['cpu_gflops'], marker='o', linewidth=2, 
            label='CPU Throughput', color='steelblue', markersize=8)
    ax.plot(df['size'], df['gpu_gflops'], marker='s', linewidth=2,
            label='GPU Throughput', color='coral', markersize=8)
    
    # Annotate peak performance
    peak_gpu_idx = df['gpu_gflops'].idxmax()
    peak_size = df.loc[peak_gpu_idx, 'size']
    peak_gflops = df.loc[peak_gpu_idx, 'gpu_gflops']
    
    ax.annotate(f'Peak: {peak_gflops:.1f} GFLOPS\n@ size {int(peak_size)}',
                xy=(peak_size, peak_gflops),
                xytext=(peak_size + 3000, peak_gflops + 5),
                fontsize=11, fontweight='bold', color='coral',
                arrowprops=dict(arrowstyle='->', color='coral', lw=2))
    
    ax.set_xlabel('Matrix Size', fontsize=13)
    ax.set_ylabel('Throughput (GFLOPS)', fontsize=13)
    ax.set_title('Computational Throughput Scaling', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/charts/throughput_scaling.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/charts/throughput_scaling.png")
    plt.close()


def plot_memory_bottleneck():
    """Visualize memory bottleneck impact."""
    df = pd.read_csv('results/benchmarks/m4_matmul_results.csv')
    
    # Calculate memory footprint (GB)
    df['memory_gb'] = (df['size']**2 * 4 * 3) / 1e9  # 3 matrices, 4 bytes each
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top: Speedup vs Memory
    color = 'tab:green'
    ax1.plot(df['memory_gb'], df['speedup'], marker='D', linewidth=3, 
             color=color, markersize=10)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.fill_between(df['memory_gb'], 0, df['speedup'], 
                      where=(df['speedup'] > 1.8), alpha=0.2, color='green')
    ax1.set_xlabel('Total Memory Footprint (GB)', fontsize=13)
    ax1.set_ylabel('GPU Speedup (×)', fontsize=13, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title('Memory Pressure Impact on Performance', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Annotate bottleneck onset
    bottleneck_idx = df[df['speedup'] < 1.5].index[0] if any(df['speedup'] < 1.5) else len(df)-1
    ax1.annotate('Bottleneck Onset',
                 xy=(df.loc[bottleneck_idx, 'memory_gb'], df.loc[bottleneck_idx, 'speedup']),
                 xytext=(df.loc[bottleneck_idx, 'memory_gb'] - 1, 2.3),
                 fontsize=11, fontweight='bold', color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Bottom: Memory vs Time
    ax2_twin = ax2.twinx()
    
    ax2.plot(df['memory_gb'], df['cpu_time_s'], marker='o', linewidth=2,
             label='CPU Time', color='steelblue', markersize=8)
    ax2.plot(df['memory_gb'], df['gpu_time_s'], marker='s', linewidth=2,
             label='GPU Time', color='coral', markersize=8)
    ax2.set_xlabel('Total Memory Footprint (GB)', fontsize=13)
    ax2.set_ylabel('Execution Time (s)', fontsize=13)
    ax2.set_yscale('log')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/charts/memory_bottleneck.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/charts/memory_bottleneck.png")
    plt.close()


def plot_efficiency_comparison():
    """Compare efficiency across operation types."""
    matmul = pd.read_csv('results/benchmarks/m4_matmul_results.csv')
    opchain = pd.read_csv('results/benchmarks/m4_operation_chain_results.csv')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot both operation types
    ax.plot(matmul['size'], matmul['speedup'], marker='o', linewidth=3,
            label='Matrix Multiplication (1 kernel)', color='darkblue', markersize=10)
    ax.plot(opchain['size'], opchain['speedup'], marker='^', linewidth=3,
            label='Operation Chain (6 kernels)', color='darkred', markersize=10)
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.axhspan(1.8, 2.5, alpha=0.1, color='green', label='Optimal Range')
    
    ax.set_xlabel('Workload Size', fontsize=13)
    ax.set_ylabel('GPU Speedup (×)', fontsize=13)
    ax.set_title('Operation Type Comparison: Impact of Kernel Count', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Annotate key insight
    ax.text(7000, 0.3, 'Operation chains hit\nmemory wall 2.5× earlier',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/charts/efficiency_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/charts/efficiency_comparison.png")
    plt.close()


if __name__ == '__main__':
    print("Generating advanced performance charts...\n")
    plot_throughput_scaling()
    plot_memory_bottleneck()
    plot_efficiency_comparison()
    print("\n✅ All advanced charts generated!")
