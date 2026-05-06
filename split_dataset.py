import os
import random
import shutil
from pathlib import Path

def split_dataset(base_dir, train_ratio=0.8):
    images_dir = Path(base_dir) / 'images'
    labels_dir = Path(base_dir) / 'labels'

    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

    random.seed(42)
    random.shuffle(image_files)

    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    for phase, files in zip(['train', 'val'], [train_files, val_files]):
        for sub in ['images', 'labels']:
            (Path(base_dir) / sub / phase).mkdir(parents=True, exist_ok=True)

        for img_path in files:
            label_path = labels_dir / (img_path.stem + '.txt')

            shutil.copy(img_path, Path(base_dir) / 'images' / phase / img_path.name)
            shutil.copy(label_path, Path(base_dir) / 'labels' / phase / label_path.name)

    print(f"✅ Done! {len(train_files)} train / {len(val_files)} val samples.")

if __name__ == "__main__":
    split_dataset("yolo/dataset")  # 可根据你路径结构调整
