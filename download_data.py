"""
Instructions and helper script for downloading the Flickr8k dataset.

The Flickr8k dataset is fairly large (~1GB of images + captions), so
automatic download in code isn't reliable (Kaggle requires login).

STEPS (do this once):
1. Create a free Kaggle account: https://www.kaggle.com
2. Go to the dataset page: https://www.kaggle.com/datasets/adityajn105/flickr8k
3. Click "Download" to get the zip file.
4. Extract the zip so you end up with this folder structure:

   data/
     Images/              <- 8000+ .jpg files
     captions.txt          <- image_name,caption format

5. Place this "data/" folder inside this project folder
   (image-caption-generator/).

Alternative: Use the Kaggle API to download from the command line:
   pip install kaggle
   # Place your kaggle.json API key in ~/.kaggle/
   # (Kaggle account settings -> "Create New API Token" to download it)
   kaggle datasets download -d adityajn105/flickr8k
   unzip flickr8k.zip -d data/

Running this script checks whether the data folder exists correctly.
"""

import os

def check_data():
    if not os.path.exists("data/Images") or not os.path.exists("data/captions.txt"):
        print("❌ data/Images folder or data/captions.txt file is missing.")
        print("   Follow the instructions above to download the Flickr8k dataset.")
        return False
    num_images = len(os.listdir("data/Images"))
    print(f"✅ Data found! {num_images} images in data/Images.")
    return True

if __name__ == "__main__":
    check_data()
