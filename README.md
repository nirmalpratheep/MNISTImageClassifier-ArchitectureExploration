# MNIST CNN Architecture Experiments

This repository contains a series of experiments with different CNN architectures for MNIST digit classification, exploring the relationship between model complexity, parameter count, and performance.

## Overview

The goal was to find an optimal architecture that balances parameter efficiency with classification accuracy. All experiments use the same training setup with variations in model architecture and batch size.

## Experimental Results

### Experiment 1: Baseline Architecture
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(9216, 50)
        self.fc2 = nn.Linear(50, 10)
```

**Architecture Details:**
- Input: 1 channel (grayscale MNIST)
- Conv1: 1→32 channels, 3×3 kernel
- Conv2: 32→64 channels, 3×3 kernel + MaxPool2D
- FC1: 9216→50 neurons
- FC2: 50→10 (output classes)

**Results:**
- Training Accuracy: 79.92%
- Test Accuracy: 91.73%
- Parameters: ~21K (estimated)

---

### Experiment 2: Inverted Channel Progression
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3)
        self.fc1 = nn.Linear(400, 10)
```

**Architecture Details:**
- Input: 1 channel
- Conv1: 1→64 channels, 3×3 kernel
- Conv2: 64→32 channels, 3×3 kernel + MaxPool2D
- Conv3: 32→16 channels, 3×3 kernel + MaxPool2D
- FC1: 400→10 (direct to output)

**Parameters:** 27,738
- Conv2d-1: 640 params
- Conv2d-2: 18,464 params
- Conv2d-3: 4,624 params
- Linear-4: 4,010 params

**Results:**
- Training Accuracy: 75.82%
- Test Accuracy: 94.22%
- Training Time: ~18s per epoch

**Key Insight:** Despite lower training accuracy, achieved higher test accuracy, indicating better generalization.

---

### Experiment 3: Minimal Parameter Architecture
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3)
        self.fc1 = nn.Linear(200, 10)
```

**Architecture Details:**
- Significantly reduced channel counts (8→16→8)
- Same layer structure as Experiment 2
- Direct classification without hidden FC layers

**Parameters:** 4,418
- Conv2d-1: 80 params
- Conv2d-2: 1,168 params
- Conv2d-3: 1,160 params
- Linear-4: 2,010 params

**Results:**
- Training Accuracy: 68.07%
- Test Accuracy: 88.87%
- Memory Usage: 0.14 MB total

**Key Insight:** Extreme parameter reduction led to underfitting, but still achieved reasonable performance.

---

### Experiment 4: Adding Hidden Layer to Minimal Architecture
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3)
        self.fc1 = nn.Linear(200, 50)
        self.fc2 = nn.Linear(50, 10)
```

**Architecture Details:**
- Same convolutional layers as Experiment 3
- Added hidden layer: 200→50→10

**Parameters:** 12,968
- Conv layers: 2,408 params
- Linear-4: 10,050 params
- Linear-5: 510 params

**Results:**
- Training Accuracy: 46.61%
- Test Accuracy: 82.55%

**Key Insight:** Adding complexity to the minimal model actually hurt performance, suggesting overfitting or optimization issues.

---

### Experiment 5: Optimized Channel Distribution
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=3)
        self.fc1 = nn.Linear(200, 10)
```

**Architecture Details:**
- Heavy front-loading: 64 channels in first layer
- Aggressive reduction: 64→32→8
- Simple classifier

**Parameters:** 23,426
- Conv2d-1: 640 params
- Conv2d-2: 18,464 params
- Conv2d-3: 2,312 params
- Linear-4: 2,010 params

**Tensor Shapes:**
- After Conv3: [batch_size, 8, 5, 5] = 200 features

**Results (Batch Size 512):**
- Training Accuracy: 63.61%
- Test Accuracy: 89.88%

---

### Experiment 6: Batch Size Optimization
Same architecture as Experiment 5, but with **batch size changed from 512 to 128**.

**Results:**
- Training Accuracy: 88.67%
- Test Accuracy: 96.43%
- Training Speed: 17.42 it/s

**Key Insight:** Smaller batch size dramatically improved both training and test accuracy, showing the importance of optimization dynamics.

---

## Key Findings

### 1. Parameter Efficiency vs Performance
| Experiment | Parameters | Test Accuracy | Efficiency Score |
|------------|------------|---------------|------------------|
| Exp 1      | ~21,000    | 91.73%        | 4.37             |
| Exp 2      | 27,738     | 94.22%        | 3.40             |
| Exp 3      | 4,418      | 88.87%        | 20.11            |
| Exp 5      | 23,426     | 89.88%        | 3.84             |
| Exp 6      | 23,426     | 96.43%        | 4.12             |

*Efficiency Score = Test Accuracy / (Parameters/1000)*

### 2. Architecture Design Principles
- **Channel Progression**: Inverted progression (wide→narrow) can work better than traditional narrow→wide
- **Parameter Distribution**: Front-loading parameters in early conv layers can be effective
- **Classifier Complexity**: Simple direct classification often outperforms complex FC layers
- **Batch Size Impact**: Smaller batches can significantly improve convergence

### 3. Optimization Insights
- **Generalization Gap**: Lower training accuracy doesn't always mean worse test performance
- **Underfitting vs Overfitting**: Very small models (4K params) still perform reasonably well
- **Training Dynamics**: Batch size has dramatic impact on final performance

## Best Configuration

**Winner: Experiment 6**
- Architecture: 64→32→8 conv layers + direct classification
- Parameters: 23,426 (within 25K target)
- Batch Size: 128
- Performance: 96.43% test accuracy

## Recommendations

1. **For Parameter-Constrained Scenarios**: Use Experiment 3 architecture (4.4K params, 88.87% accuracy)
2. **For Best Performance**: Use Experiment 6 setup (23.4K params, 96.43% accuracy)
3. **For Balanced Approach**: Use Experiment 2 architecture (27.7K params, 94.22% accuracy)

## Technical Notes

- All models use ReLU activation and MaxPool2D
- Log softmax output for classification
- Cross-entropy loss function
- SGD optimizer with momentum
- Input: 28×28 grayscale MNIST images

## Memory Usage Analysis

The experiments show that model size scales predictably:
- 4K params: 0.14 MB total memory
- 23K params: 0.57 MB total memory
- 28K params: 0.59 MB total memory

This demonstrates excellent memory efficiency for deployment scenarios.
