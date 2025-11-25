import os
import pandas as pd

def create_partitions(data_dir, output_dir):
    classes = ['rock', 'paper', 'scissors']
    partitions = {'train': [], 'devtest': [], 'test': []}

    # Logic extracted from your notebook
    for cls in classes:
        class_dir = os.path.join(data_dir, cls)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found.")
            continue

        images = os.listdir(class_dir)
        # Simple split logic (matches your notebook: 50/25/25 approx)
        # You can make this smarter e.g. random shuffle, stratified sampling, etc.
        for i, img_name in enumerate(images):
            img_path = os.path.join(cls, img_name) # Relative path
            if i < 50:
                partitions['train'].append({'path': img_path, 'label': cls})
            elif i < 75:
                partitions['devtest'].append({'path': img_path, 'label': cls})
            else:
                partitions['test'].append({'path': img_path, 'label': cls})

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    for partition_name, data in partitions.items():
        df = pd.DataFrame(data)
        csv_path = os.path.join(output_dir, f"{partition_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {partition_name}.csv with {len(df)} images.")

if __name__ == "__main__":
    # Default assumes running from project root
    raw_data_path = os.path.join("data", "raw")
    csv_output_path = os.path.join("data", "partitions")
    create_partitions(raw_data_path, csv_output_path)