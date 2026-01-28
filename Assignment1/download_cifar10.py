import requests
import tarfile
import os

def download_file(url, save_path):
    print(f"Downloading {url} to {save_path}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

def extract_tar_gz(file_path, extract_path):
    print(f"Extracting {file_path} to {extract_path}...")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print("Extraction complete.")

def main():
    base_dir = r"c:/Users/Dell/Desktop/AI_LEARNING/Assignment1"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(base_dir, "cifar-10-python.tar.gz")
    extract_dir = base_dir
    
    # Check if already extracted
    if os.path.exists(os.path.join(base_dir, "cifar-10-batches-py")):
        print("CIFAR-10 dataset already exists.")
        return

    download_file(url, tar_path)
    extract_tar_gz(tar_path, extract_dir)
    
    # Clean up tar file
    if os.path.exists(tar_path):
        os.remove(tar_path)
        print("Removed tar file.")

if __name__ == "__main__":
    main()
