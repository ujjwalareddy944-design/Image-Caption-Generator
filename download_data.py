"""
Flickr8k dataset download cheyadaniki instructions + helper script.

Flickr8k dataset chala pedadi (~1GB images + captions), so direct code tho
automatic download cheyadam reliable ga undadu (Kaggle login kavali).

STEPS (manual, oka sari chesthe chalu):
1. Kaggle account create cheyandi (free): https://www.kaggle.com
2. Ee dataset page ki వెళ్ళండి: https://www.kaggle.com/datasets/adityajn105/flickr8k
3. "Download" button click chesi zip file download cheyandi.
4. Zip ni extract chesi, ee folder structure vachela chudandi:

   data/
     Images/              <- 8000+ .jpg files
     captions.txt          <- image_name,caption format lo undi

5. Ee "data/" folder ni ee project folder (1-image-caption-generator/) lo
   pettandi.

Alternative: Kaggle API vadi command line nundi download cheyachu:
   pip install kaggle
   # kaggle.json API key ni ~/.kaggle/ folder lo pettandi (Kaggle account
   # settings -> Create New API Token nundi download avutundi)
   kaggle datasets download -d adityajn105/flickr8k
   unzip flickr8k.zip -d data/

Ee script run chesthe, data folder undo ledo check chestundi.
"""

import os

def check_data():
    if not os.path.exists("data/Images") or not os.path.exists("data/captions.txt"):
        print("❌ data/Images folder leda data/captions.txt file missing.")
        print("   Pైన instructions follow chesi Flickr8k dataset download cheyandi.")
        return False
    num_images = len(os.listdir("data/Images"))
    print(f"✅ Data found! {num_images} images unnayi data/Images lo.")
    return True

if __name__ == "__main__":
    check_data()
