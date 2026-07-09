import os
from PIL import Image

INPUT_FOLDER = "yellow_raws"      # folder with your original images
OUTPUT_FOLDER = "yellow_raws"        # resized + converted images go here
TARGET_SIZE = (640, 640)                # width, height — YOLO's default input size

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif")
count = 0

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(valid_ext):
        img_path = os.path.join(INPUT_FOLDER, filename)
        img = Image.open(img_path).convert("RGB")   # convert handles PNG transparency, grayscale, etc.
        img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)

        # force .jpg extension regardless of original format
        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(OUTPUT_FOLDER, base_name + ".jpg")

        img_resized.save(save_path, "JPEG", quality=95)
        count += 1

print(f"Converted + resized {count} images to {TARGET_SIZE} JPG → saved in '{OUTPUT_FOLDER}'")