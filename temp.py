import os
import shutil
import random

# Paths
data_dir = "data/raw"
train_dir = "data/train"
val_dir = "data/val"
classes = ["Ayrshire cattle", "Brown Swiss cattle", "Holstein Friesian cattle", "Jersey cattle", "Red Dane cattle"]

# Make folders
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

# Split 80% train, 20% val
for cls in classes:
    cls_files = [f for f in os.listdir(os.path.join(data_dir, cls)) if f.endswith(".jpg")]
    random.shuffle(cls_files)
    split_idx = int(0.8 * len(cls_files))
    
    for f in cls_files[:split_idx]:
        shutil.copy(os.path.join(data_dir, cls, f), os.path.join(train_dir, cls, f))
    for f in cls_files[split_idx:]:
        shutil.copy(os.path.join(data_dir, cls, f), os.path.join(val_dir, cls, f))
