import os
from PIL import Image

def resize_images(directories, size=(256, 256)):
    for directory in directories:
        # Get the full path of the directory
        dir_path = os.path.join(os.getcwd(), directory)
        
        # Get the list of files in the directory
        files = os.listdir(dir_path)
        
        for file in files:
            file_path = os.path.join(dir_path, file)
            
            # Check if the file is an image
            try:
                with Image.open(file_path) as img:
                    # Resize the image
                    img = img.resize(size, Image.Resampling.LANCZOS)
                    # Save the resized image back to the same path
                    img.save(file_path)
                print(f"Resized {file_path}")
            except IOError:
                print(f"Skipping non-image file {file_path}")

def rename_images(directories):
    for directory in directories:
        # Get the full path of the directory
        dir_path = os.path.join(os.getcwd(), directory)
        
        # Get the list of image files in the directory
        files = sorted(os.listdir(dir_path))
        
        img_num = 1
        for file in files:
            file_path = os.path.join(dir_path, file)
            
            # Check if the file is an image
            try:
                with Image.open(file_path) as img:
                    # Define the new file name
                    new_file_name = f"{img_num}.png"
                    new_file_path = os.path.join(dir_path, new_file_name)
                    
                    # Rename the file
                    os.rename(file_path, new_file_path)
                    img_num += 1
                print(f"Renamed {file_path} to {new_file_path}")
            except IOError:
                print(f"Skipping non-image file {file_path}")

def delete_corrupt_images(directories):
    for directory in directories:
        dir_path = os.path.join(os.getcwd(), directory)
        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} does not exist, skipping.")
            continue
        
        files = os.listdir(dir_path)
        if not files:
            print(f"Directory {dir_path} is empty, skipping.")
            continue
        
        for file in files:
            file_path = os.path.join(dir_path, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify the image file is not corrupt
                print(f"Verified {file_path}")
            except (IOError, SyntaxError) as e:
                print(f"Deleting corrupt image file {file_path}")
                os.remove(file_path)


# if __name__ == "__main__":
#     directories = ['dir1', 'dir2', 'dir3']  # Replace with your directories

#     # Delete small images
#     delete_small_images(directories)

#     # Resize images
#     resize_images(directories)

#     # Rename images
#     rename_images(directories)

if __name__ == "__main__":
    directories = [
        'BrainScanData/train/glioma', 
        'BrainScanData/train/meningioma', 
        'BrainScanData/train/notumor',
        'BrainScanData/train/pituitary',
        'BrainScanData/test/glioma', 
        'BrainScanData/test/meningioma', 
        'BrainScanData/test/notumor',
        'BrainScanData/test/pituitary'
        ]  # Replace with your directories
    resize_images(directories)
    rename_images(directories)
    delete_corrupt_images(directories)
