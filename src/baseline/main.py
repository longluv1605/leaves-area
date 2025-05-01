import yaml
from pathlib import Path
from typing import List
from plotting import show_images
from segment import compute_area

def main() -> None:
    """Process multiple images to compute leaf areas and save results.

    Reads configuration from config.yaml, processes images, and saves visualizations.
    """
    # Load configuration
    config_path = Path('./src/baseline/config.yaml')
    try:
        with config_path.open('r') as file:
            config = yaml.safe_load(file)
            if not config or 'RESULTS' not in config:
                raise ValueError("Invalid or empty configuration file")
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

    # Get save directory and ensure it exists
    save_dir = Path(config['RESULTS']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    image_names = config['RESULTS'].get('image_paths', [f'im{i}.jpg' for i in range(1, 4)])
    for idx, image_name in enumerate(image_names, 1):
        image_path = Path('./') / image_name
        try:
            result = compute_area(str(image_path))
            images = result['images']
            labels = result['labels']
            save_path = save_dir / f'result{idx}.png'
            show_images(
                images[-2:],
                labels[-2:],
                plot_size=(1, 2),
                figsize=(8, 14),
                save_path=str(save_path)
            )
            print(f"Processed and saved result for {image_name} to {save_path}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error processing {image_name}: {e}")

if __name__ == '__main__':
    main()