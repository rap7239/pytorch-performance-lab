# PyTorch Performance Lab

**Exploring GPU acceleration, performance optimization, and gradient-based learning on Apple Silicon (M4 Mini)**

## 🎯 Project Overview

This repository documents a systematic exploration of PyTorch performance characteristics and machine learning fundamentals, with a focus on:
- CPU vs GPU performance analysis
- Identifying performance sweet spots and bottlenecks
- Understanding gradient descent and automatic differentiation
- Building foundation for production ML performance engineering

**Hardware:** Mac M4 Mini with MPS (Metal Performance Shaders) GPU acceleration

---

## 📊 Key Findings

### Performance Regimes Discovered

Through systematic benchmarking, I identified three distinct performance regimes when comparing CPU vs GPU execution:

#### 1. **Overhead Regime** (Small Workloads)
- **Matrix Size:** < 2,000 × 2,000
- **Observation:** GPU slower than CPU
- **Cause:** Data transfer overhead exceeds computation benefit
- **Speedup:** 0.4x - 0.8x (GPU slower)

#### 2. **Sweet Spot** (Medium Workloads)
- **Matrix Size:** 5,000 × 8,000
- **Observation:** GPU significantly faster
- **Cause:** Parallel computation benefit > overhead
- **Speedup:** 2.0x - 2.2x (GPU faster)

#### 3. **Memory Bottleneck** (Large Workloads)
- **Matrix Size:** > 10,000 × 10,000
- **Observation:** GPU advantage diminishes
- **Cause:** Memory bandwidth saturation
- **Speedup:** 1.0x - 1.5x (advantage shrinking)

### Operation-Specific Performance

Different operations exhibit different performance profiles:

| Operation Type | Sweet Spot Size | Peak Speedup | Bottleneck Onset |
|---------------|-----------------|--------------|------------------|
| Matrix Multiplication | 5000-8000 | 2.2x | 20,000 |
| Operation Chains (normalize→square→sum) | ~5000 | 3.5x | 8,000 |
| Element-wise Operations | N/A | 0.6x | Immediate |

**Key Insight:** Not all operations benefit from GPU acceleration. Profiling is essential.

---

## 📁 Repository Contents

### Notebooks

1. **[01_tensor_fundamentals.ipynb](notebooks/01_tensor_fundamentals.ipynb)**
   - Tensor creation, indexing, slicing
   - Shape manipulation and reshaping
   - GPU tensor operations
   - Mental model: "boxes within boxes"

2. **[02_gpu_benchmarking.ipynb](notebooks/02_gpu_benchmarking.ipynb)**
   - Systematic CPU vs GPU performance comparison
   - Matrix multiplication benchmarks
   - Operation chain benchmarks
   - Performance regime identification

3. **[03_gradients_and_learning.ipynb](notebooks/03_gradients_and_learning.ipynb)**
   - Automatic differentiation fundamentals
   - Gradient descent algorithm implementation
   - Learning rate impact analysis
   - Visualization of learning curves

4. **[04_multi_parameter_learning.ipynb](notebooks/04_multi_parameter_learning.ipynb)**
   - Linear regression from scratch
   - Multiple parameter optimization
   - Underdetermined vs well-determined systems

### Documentation

- **[performance_regimes.md](docs/performance_regimes.md)** - Detailed analysis of performance characteristics
- **[learning_notes.md](docs/learning_notes.md)** - Key concepts and insights

### Results

- **charts/** - Performance visualization plots
- **benchmarks/** - Raw benchmark data (CSV format)

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.11+
PyTorch 2.8+ with MPS support
Jupyter Notebook
matplotlib
```

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/pytorch-performance-lab.git
cd pytorch-performance-lab

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Running Benchmarks

```python
from utils.benchmark_tools import benchmark_matmul

# Benchmark matrix multiplication at different sizes
sizes = [1000, 2000, 5000, 8000, 10000]
results = benchmark_matmul(sizes)
```

---

## 📈 Sample Results

### Matrix Multiplication Performance (M4 Mini)

| Size | CPU Time | GPU Time | Speedup |
|------|----------|----------|---------|
| 1000 | 0.0015s | 0.0019s | 0.79x |
| 2000 | 0.0145s | 0.0075s | 1.93x |
| 5000 | 0.1726s | 0.0796s | 2.17x ⭐ |
| 8000 | 0.6716s | 0.3359s | 2.00x |
| 10000 | 1.2671s | 0.7067s | 1.79x |

⭐ = Optimal performance region

---

## 🎓 Learning Outcomes

Through this exploration, I gained hands-on experience with:

✅ **Performance Engineering**
- Identifying computational bottlenecks
- Understanding overhead vs computation trade-offs
- Hardware-aware optimization

✅ **Machine Learning Fundamentals**
- Automatic differentiation
- Gradient descent optimization
- Loss functions and convergence

✅ **PyTorch Proficiency**
- Tensor operations and GPU acceleration
- Training loop implementation
- Performance profiling techniques

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Framework:** PyTorch 2.8
- **Hardware:** Apple M4 Mini (MPS GPU)
- **Tools:** Jupyter Notebook, matplotlib, NumPy

---

## 📝 Future Work

- [ ] Extend benchmarks to larger models (neural networks)
- [ ] Compare different optimization algorithms (SGD, Adam, AdamW)
- [ ] Profile memory usage patterns
- [ ] Implement mixed-precision training benchmarks
- [ ] Build automated benchmark dashboard

---

## 👤 Author

RAHUL PUNDE
- Building toward ML performance engineering
- Focus: Production-scale model optimization and observability

---

## 📄 License

MIT License - feel free to use this code for learning and experimentation.

---

## 🙏 Acknowledgments

Exploring PyTorch performance characteristics through systematic experimentation and documentation.
