import os
import shutil
import random

def split_dataset(dataset_dir, train_ratio=0.8):
    # Set seed for reproducibility
    random.seed(42)
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} does not exist.")
        return

    # List all entries in dataset_dir
    entries = os.listdir(dataset_dir)
    
    # Exclude 'train' and 'test' from categories
    categories = [
        entry for entry in entries 
        if os.path.isdir(os.path.join(dataset_dir, entry)) 
        and entry.lower() not in ['train', 'test']
    ]
    
    print(f"Found {len(categories)} animal categories: {categories}")
    
    # Paths for train and test
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    total_train_count = 0
    total_test_count = 0
    
    for category in categories:
        category_path = os.path.join(dataset_dir, category)
        
        # Get all files in category directory (excluding subdirectories if any)
        files = [
            f for f in os.listdir(category_path) 
            if os.path.isfile(os.path.join(category_path, f))
        ]
        
        # Shuffle files
        random.shuffle(files)
        
        # Calculate split index
        split_idx = int(len(files) * train_ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        # Create output directories
        category_train_dir = os.path.join(train_dir, category)
        category_test_dir = os.path.join(test_dir, category)
        
        os.makedirs(category_train_dir, exist_ok=True)
        os.makedirs(category_test_dir, exist_ok=True)
        
        print(f"Processing category '{category}': total {len(files)} files.")
        print(f"  -> Moving {len(train_files)} files to train...")
        for f in train_files:
            src = os.path.join(category_path, f)
            dst = os.path.join(category_train_dir, f)
            shutil.move(src, dst)
            
        print(f"  -> Moving {len(test_files)} files to test...")
        for f in test_files:
            src = os.path.join(category_path, f)
            dst = os.path.join(category_test_dir, f)
            shutil.move(src, dst)
            
        total_train_count += len(train_files)
        total_test_count += len(test_files)
        
        # Clean up the original empty category directory
        try:
            os.rmdir(category_path)
            print(f"Deleted empty category folder: {category_path}")
        except Exception as e:
            print(f"Could not delete folder {category_path}: {e}")
            
    print("\nDataset split complete!")
    print(f"Total training images: {total_train_count}")
    print(f"Total testing images: {total_test_count}")
    print(f"Grand total: {total_train_count + total_test_count}")

if __name__ == "__main__":
    # Base directory of the project
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'Dataset')
    split_dataset(dataset_dir)
