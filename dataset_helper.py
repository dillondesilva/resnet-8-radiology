import os
import shutil
import random

def split_images(source_folder, train_folder, test_folder, train_ratio=0.8):
    # Create target directories if they don't exist
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    
    # Get all files from the source folder
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # Shuffle the list of files
    random.shuffle(all_files)
    
    # Calculate the split point
    split_point = int(len(all_files) * train_ratio)
    
    # Split the files into train and split sets
    train_files = all_files[:split_point]
    test_files = all_files[split_point:]
    
    # Move the files to the respective directories
    for file_name in train_files:
        shutil.move(os.path.join(source_folder, file_name), os.path.join(train_folder, file_name))
    
    for file_name in test_files:
        shutil.move(os.path.join(source_folder, file_name), os.path.join(test_folder, file_name))
    
    print(f"Moved {len(train_files)} files to {train_folder}")
    print(f"Moved {len(test_files)} files to {test_folder}")


source_folder = './Data/glioma_tumor'
train_folder = './Data/glioma_tumor/train'
split_folder = './Data/glioma_tumor/test'
split_images(source_folder, train_folder, split_folder, train_ratio=0.9)

source_folder = './Data/meningioma_tumor'
train_folder = './Data/meningioma_tumor/train'
split_folder = './Data/meningioma_tumor/test'
split_images(source_folder, train_folder, split_folder, train_ratio=0.9)

source_folder = './Data/normal'
train_folder = './Data/normal/train'
split_folder = './Data/normal/test'
split_images(source_folder, train_folder, split_folder, train_ratio=0.9)

source_folder = './Data/pituitary_tumor'
train_folder = './Data/pituitary_tumor/train'
split_folder = './Data/pituitary_tumor/test'
split_images(source_folder, train_folder, split_folder, train_ratio=0.9)
