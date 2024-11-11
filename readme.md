# Vision Transformer (ViT) Implementation

This repository contains a Vision Transformer (ViT) model implementation using PyTorch and `einops`. You can train, evaluate, or run inference on the CIFAR-10 dataset or perform image classification with pre-trained or custom-trained weights.

## Usage

The `vit.py` script supports the following modes:

- **Train the Model**: Train on CIFAR-10.
- **Evaluate the Model**: Test accuracy on CIFAR-10.
- **Inference with Pre-trained Weights**: Classify an image using Hugging Face weights.
- **Inference with Custom Weights**: Classify an image using your trained model.

### Commands

- **Train**: Train the ViT model on CIFAR-10 for 20 epochs (customizable).

  ```bash
  python vit.py --train
  ```

- **Evaluate**: Test accuracy of the trained model on CIFAR-10.

  ```bash
  python vit.py --evaluate
  ```

- **Inference with Pre-trained Weights**: Classify an image.

  ```bash
  python vit.py --image_path /path/to/image.jpg
  ```

- **Inference with Custom Weights**: Classify an image with custom-trained weights.

  ```bash
  python vit.py --use-trained-weights --image_path /path/to/image.jpg
  ```

### Default Behavior

Running `vit.py` without arguments defaults to inference with pre-trained weights and a sample dog image from PyTorch.

## Directory Structure

- `vit.py`: Main script.
- `checkpoints/`: Saved model checkpoints.
- `logs/`: Training and inference logs.
- `data/`: Dataset storage (auto-created).

## Notes

- **Image Formats**: Use JPG or PNG.
- **GPU Support**: Automatically uses GPU if available.
- **Hyperparameters**: Adjustable in the script.
- **Custom Datasets**: For other datasets (e.g., CIFAR-100), update dataset loading and `num_classes` in the script.

For detailed usage examples:

```bash
python vit.py --train
python vit.py --evaluate
python vit.py --image_path /path/to/image.jpg
python vit.py --use-trained-weights --image_path /path/to/image.jpg
```
