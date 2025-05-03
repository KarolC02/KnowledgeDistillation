from PIL import Image
import os

dataset_root = "datasets/rp2k/train"

problem_images = []

for root, _, files in os.walk(dataset_root):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                if img.mode == "P" and "transparency" in img.info:
                    problem_images.append(path)
            except Exception as e:
                print(f"Failed to load {path}: {e}")

print(f"Found {len(problem_images)} problematic images:")
for path in problem_images:
    print(path)