# DeepLabV3+ Semantic Segmentation

This repository implements the DeepLabV3+ model for semantic segmentation using PyTorch. The model is designed for flexibility and portability across different systems.

## Project Structure

```plain
leaves-area/
└── src/
    └── dl_method/
        ├── models/
        │   └── deeplab.py
        ├── utils/
        │   ├── data.py
        │   ├── loss.py
        │   ├── training.py
        │   └── visualize.py
        ├── config.yaml
        └── main.py
```

## Installation

1. Clone the repository:

    ```bash
    git clone <repository_url>
    cd repo
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Dependencies

- Python >= 3.8
- PyTorch >= 2.0
- torchvision >= 0.15
- albumentations >= 1.4.0
- opencv-python >= 4.5.0
- torchmetrics >= 1.0.0
- tqdm >= 4.66.0
- matplotlib >= 3.7.0
- numpy >= 1.24.0
- pyyaml >= 6.0

See `requirements.txt` for a complete list.

## Usage

The `src/dl_method/main.py` script trains a DeepLabV3+ model on a spinach segmentation dataset, evaluates performance, and visualizes results. It uses early stopping to prevent overfitting and saves the best model and visualization results. Hyperparameters, data augmentations, and loss settings are configured via `src/dl/config.yaml`.

### Directory Setup

Ensure your dataset is organized as follows:

```plain
datasets/spinach/
├── images/
│   ├── train/
│   └── val/
├── masks/
    ├── train/
    └── val/
```

Place new images for segmentation in:

```plain
images/
├── im1.jpg
├── im2.jpg
├── im3.jpg
```

### Configuration

Edit `src/dl_method/config.yaml` to adjust settings:

```yaml
data_dir: datasets/spinach
results_dir: results/deeplabv3plus
models_dir: save/models
batch_size: 4
num_epochs: 1
learning_rate: 0.0001
num_classes: 3
num_workers: 4
transformations:
  horizontal_flip_p: 0.5
  random_rotate_p: 0.5
  elastic_transform_p: 0.3
  grid_distortion_p: 0.3
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
loss:
  smooth: 0.00001
  class_weights: [0.1, 1.0, 5.0]
  dice_ce_weights: [0.5, 0.5]
  ignore_index: null
```

### Running the Script

```bash
python src/dlmain.py
```

This will:

1. Train the model for up to 1 epoch (configurable in config.yaml).
2. Evaluate training and validation metrics (loss, IoU, accuracy).
3. Save training metrics plot to `results/deeplabv3plus/training_metrics.png`.
4. Save the best model to `save/models/model_<timestamp>.pth`.
5. Visualize segmentation results for validation images in `results/deeplabv3plus/val_<index>.png`.
6. Visualize segmentation for new images in `results/deeplabv3plus/new_image_<index>.png`.

## Development Standards

Code follows PEP 8 guidelines.
Type hints are used for better code clarity.
Docstrings document all classes, methods, and functions.
The project is structured for portability and modularity.

## License

This project is licensed under the MIT License. See LICENSE for details.
