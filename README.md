# MNIST CNN Architecture Exploration 🧠

A comprehensive study of Convolutional Neural Network architectures for MNIST digit classification, exploring the trade-offs between accuracy, parameter efficiency, and training stability.

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture Experiments](#architecture-experiments)
- [Key Results](#key-results)
- [Best Performing Architecture](#best-performing-architecture)
- [Performance Analysis](#performance-analysis)
- [Recommendations](#recommendations)
- [Training Configuration](#training-configuration)
- [Getting Started](#getting-started)

## 🎯 Overview

This repository contains 5 distinct CNN architecture experiments on the MNIST dataset, ranging from simple baseline models to deep networks with batch normalization. Each experiment focuses on different architectural principles to understand their impact on model performance and efficiency.

**Dataset**: MNIST handwritten digits (28×28 grayscale images)  
**Task**: 10-class classification  
**Metrics**: Test accuracy, parameter count, training stability

## 🔬 Architecture Experiments

| Experiment | Architecture | Parameters | Test Accuracy | Key Innovation |
|------------|-------------|------------|---------------|----------------|
| **Exp 1** | Baseline CNN | 21,000 | 91.73% | Simple 2-conv + FC baseline |
| **Exp 2** | Wide-to-Narrow | 27,700 | 94.22% | Inverted channel progression |
| **Exp 3** | Optimized Channels | 23,400 | 96.43% | 64→32→8 progression + small batch |
| **Exp 4** | Deep CNN | 17,054 | 99.25% | 7 convolutional layers |
| **Exp 5** | Deep + BatchNorm | 17,302 | **99.50%** | BatchNorm + deep architecture |

## 🏆 Key Results

### Performance Progression
```
Exp 1: 91.73% (21K params) → Baseline
Exp 2: 94.22% (27.7K)      → +2.49% with channel optimization
Exp 3: 96.43% (23.4K)      → +2.21% with batch size tuning
Exp 4: 99.25% (17K)        → +2.82% with depth increase
Exp 5: 99.50% (17.3K)      → +0.25% with BatchNorm
```

### Efficiency Analysis
- **Most Accurate**: Exp 5 (99.50%) - Deep CNN + BatchNorm
- **Best Parameter Efficiency**: Exp 4 (99.25% with only 17K parameters)
- **Balanced Performance**: Exp 3 (96.43% with good efficiency)

## 🏗️ Best Performing Architecture

**Experiment 5: Deep CNN + BatchNorm (99.50% accuracy)**

```python
Architecture Summary:
├── Conv2d(1→32, 3×3) + BatchNorm + ReLU     # 320 + 64 params
├── Conv2d(32→16, 3×3) + BatchNorm + ReLU    # 4,624 + 32 params
├── Conv2d(16→16, 3×3) + BatchNorm + ReLU    # 2,320 + 32 params
├── MaxPool2d(2×2)
├── Conv2d(16→16, 3×3) + BatchNorm + ReLU    # 2,320 + 32 params
├── Conv2d(16→16, 3×3) + BatchNorm + ReLU    # 2,320 + 32 params
├── Conv2d(16→16, 3×3) + BatchNorm + ReLU    # 2,320 + 32 params
├── Conv2d(16→12, 3×3) + BatchNorm + ReLU    # 1,740 + 24 params
└── Linear(108→10)                           # 1,090 params

Total Parameters: 17,302
Memory Footprint: 0.69 MB
```

### Layer-wise Output Shapes
| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| Conv2d-1 | [1, 28, 28] | [32, 26, 26] | 320 |
| BatchNorm2d-2 | [32, 26, 26] | [32, 26, 26] | 64 |
| Conv2d-3 | [32, 26, 26] | [16, 24, 24] | 4,624 |
| BatchNorm2d-4 | [16, 24, 24] | [16, 24, 24] | 32 |
| Conv2d-5 | [16, 24, 24] | [16, 22, 22] | 2,320 |
| BatchNorm2d-6 | [16, 22, 22] | [16, 22, 22] | 32 |
| MaxPool2d-7 | [16, 22, 22] | [16, 9, 9] | 0 |
| Conv2d-8 | [16, 9, 9] | [16, 7, 7] | 2,320 |
| BatchNorm2d-9 | [16, 7, 7] | [16, 7, 7] | 32 |
| Conv2d-10 | [16, 7, 7] | [16, 3, 3] | 2,320 |
| BatchNorm2d-11 | [16, 3, 3] | [16, 3, 3] | 32 |
| Conv2d-12 | [16, 3, 3] | [12, 3, 3] | 1,740 |
| BatchNorm2d-13 | [12, 3, 3] | [12, 3, 3] | 24 |
| Linear-14 | [108] | [10] | 1,090 |

## 📊 Performance Analysis

### Accuracy vs Parameters Visualization
```
Accuracy (%)
    100 ┤                            ●── Exp 5 (99.50%)
     99 ┤                       ●────────── Exp 4 (99.25%)
     98 ┤                   
     97 ┤               
     96 ┤           ●───────────────────── Exp 3 (96.43%)
     95 ┤       
     94 ┤     ●─────────────────────────── Exp 2 (94.22%)
     93 ┤
     92 ┤ ●───────────────────────────────── Exp 1 (91.73%)
     91 ┤
        └─────────────────────────────────────────────────
         15K    20K    25K    30K  Parameters
```

### Key Insights
1. **Depth matters**: Moving from 2 to 7 convolutional layers dramatically improved accuracy (+7.52%)
2. **BatchNorm helps**: Adding batch normalization provided consistent training and slight accuracy boost
3. **Parameter efficiency**: Deeper networks achieved better results with fewer parameters
4. **Channel progression**: Strategic channel reduction (wide→narrow) improved feature extraction

## 💡 Recommendations

### For Different Use Cases:

**🎯 Maximum Accuracy**
- **Use**: Experiment 5 (Deep CNN + BatchNorm)
- **Accuracy**: 99.50%
- **Trade-off**: Slightly more parameters (17.3K)

**⚡ Balanced Performance**
- **Use**: Experiment 4 (Deep CNN)
- **Accuracy**: 99.25%
- **Advantage**: Most parameter-efficient high-accuracy model

**🚀 Quick Prototyping**
- **Use**: Experiment 3 (Optimized Channels)
- **Accuracy**: 96.43%
- **Advantage**: Good balance of accuracy and simplicity

**📱 Memory Constrained**
- **Use**: Experiment 4
- **Reason**: High accuracy (99.25%) with minimal memory footprint

## ⚙️ Training Configuration

```python
# Common Training Parameters
Input Shape: (1, 28, 28)
Batch Size: 32-128 (varied by experiment)
Epochs: 10-20
Learning Rate: 0.01
Momentum: 0.9

# Architecture Components
Activation: ReLU
Pooling: MaxPool2d(2x2)
Normalization: BatchNorm2d (Exp 5)
Loss Function: CrossEntropyLoss
Optimizer: SGD with momentum
Output Activation: LogSoftmax
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision numpy matplotlib
```

### Quick Start
```python
# Clone and run experiment
git clone https://github.com/yourusername/MNISTImageClassifier-ArchitectureExploration

cd MNISTImageClassifier-ArchitectureExploration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ Star this repo if you found it helpful!**
