# Vision Transformer from Scratch

A complete implementation of the Vision Transformer (ViT) architecture from scratch using PyTorch, trained on the CIFAR-10 dataset.

## 📖 Overview

This project implements the Vision Transformer architecture as described in the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al. The implementation includes:

- **Patch Embedding**: Converts input images into patch embeddings
- **Position Embeddings**: Adds positional information to patches
- **Multi-Head Self-Attention**: Core transformer mechanism
- **MLP Blocks**: Feed-forward networks with GELU activation
- **Classification Head**: Final layer for image classification

## 🏗️ Architecture

### Key Components

1. **Patch Embedding Layer**

   - Divides input images into fixed-size patches (4x4 pixels)
   - Projects patches to embedding dimension using Conv2d
   - Adds learnable CLS token for classification
   - Includes learnable position embeddings

2. **Transformer Encoder**

   - Multi-head self-attention mechanism
   - Layer normalization
   - MLP blocks with GELU activation
   - Residual connections

3. **Classification Head**
   - Uses CLS token for final classification
   - Linear projection to number of classes

### Hyperparameters

```python
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 3e-4
PATCH_SIZE = 4
NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNELS = 3
EMBED_DIM = 256
NUM_HEADS = 8
DEPTH = 6
MLP_DIM = 512
DROP_RATE = 0.1
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- pandas
- matplotlib

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Vision-Transformer-from-scratch
```

2. Install dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib
```

### Usage

#### Running the Main Script

```bash
python main.py
```

#### Using the Jupyter Notebook

```bash
jupyter notebook Building_Vision_Transformer_on_CiFar10.ipynb
```

## 📊 Dataset

This implementation uses the **CIFAR-10** dataset, which contains:

- 60,000 32x32 color images
- 10 different classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- 50,000 training images and 10,000 test images

### Data Preprocessing

- Images are normalized using `transforms.Normalize((0.5), (0.5))`
- Converted to tensors using `transforms.ToTensor()`
- Batch size: 128 for training and testing

## 🔧 Implementation Details

### Patch Embedding

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        # Convolutional projection of patches
        # CLS token for classification
        # Position embeddings
```

### MLP Block

```python
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop_rate):
        # Two linear layers with GELU activation
        # Dropout for regularization
```

### Training Process

1. **Data Loading**: CIFAR-10 dataset with transformations
2. **Model Initialization**: Vision Transformer with specified hyperparameters
3. **Training Loop**:
   - Forward pass through the model
   - Loss computation (CrossEntropyLoss)
   - Backward pass and optimization
   - Validation on test set
4. **Evaluation**: Accuracy calculation on test dataset

## 📈 Expected Results

With the current hyperparameters:

- **Training Time**: ~10 epochs
- **Expected Accuracy**: 60-80% on CIFAR-10 (depending on training)
- **Model Size**: ~1-2M parameters

## 🛠️ Customization

### Modifying Hyperparameters

You can easily modify the hyperparameters in the main script:

```python
# Adjust these values for different experiments
BATCH_SIZE = 64  # Smaller for memory constraints
EPOCHS = 20      # More epochs for better performance
LEARNING_RATE = 1e-4  # Different learning rate
PATCH_SIZE = 8   # Larger patches
EMBED_DIM = 512  # Larger embedding dimension
```

### Using Different Datasets

To use a different dataset, modify the data loading section:

```python
# Example for MNIST
train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Original Vision Transformer paper by Dosovitskiy et al.
- PyTorch team for the excellent deep learning framework
- CIFAR-10 dataset creators

## 📚 References

1. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." arXiv preprint arXiv:2010.11929 (2020).
2. Vaswani, A., et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

## 📞 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This is a learning implementation. For production use, consider using established libraries like `timm` or `transformers` which provide optimized Vision Transformer implementations.
