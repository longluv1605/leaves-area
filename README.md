# DeepLabV3+ Semantic Segmentation

This repository implements the DeepLabV3+ model for semantic segmentation using PyTorch. The model is designed for flexibility and portability across different systems.

## Project Structure

```plain
leaves-area/
├── datasets/
│   └── spinach/
│       ├── images/
│       │   ├── train/
│       │   │   ├── train_image.jpg
│       │   │   └── ...
│       │   └── val/
│       │       ├── val_image.jpg
│       │       └── ...
│       └── masks/
│           ├── train/
│           │   ├── train_image_mask.png
│           │   └── ...
│           └── val/
│               ├── val_image_mask.png
│               └── ...
├── images/
│   ├── im1.jpg
│   ├── im2.jpg
│   └── ...
├── results/
│   ├── baseline/
│   │   └── *.png
│   └── deeplabv3plus/
│       └── *.png
├── save/
│   ├── hyperparams/
│   │   └── *.pth
│   └── models/
│       └── deeplabv3plus.pth
├── src/
│   ├── baseline/
│   │   ├── main.py
│   │   ├── config.yaml
│   │   ├── plotting.py
│   │   └── segment.py
│   └── dl_method/
│       ├── models/
│       │   └── deeplab.py
│       ├── utils/
│       │   ├── data.py
│       │   ├── loss.py
│       │   ├── training.py
│       │   └── visualize.py
│       ├── config.yaml
│       └── main.py
├── .gitignore
├── README.MD
├── requirements.txt
├── data_analyst.ipynb
├── deeplabv3plus*.ipynb
└── baseline.ipynb
```

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/longluv1605/leaves-area
    cd leaves-area
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

The `src/dl_method/main.py` script trains a DeepLabV3+ model on a spinach segmentation dataset, evaluates performance, and visualizes results. It uses early stopping to prevent overfitting and saves the best model and visualization results. Hyperparameters, data augmentations, and loss settings are configured via `src/dl_method/config.yaml`.

### Running the Script

```bash
python src/dl_method/main.py
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
