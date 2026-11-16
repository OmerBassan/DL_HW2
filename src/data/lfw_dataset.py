import os
import random
import glob
from PIL import Image
import torch
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class PairListDataset(Dataset):
    """Fixed, deterministic pairs for validation/test: item idx is always pair idx.

    pairs: list of (name1, num1, name2, num2, label), e.g. from
    src.data.splits.read_pairs(). A missing image raises instead of being
    silently replaced, because a substituted pair would change the eval set.
    """

    def __init__(self, root_dir, pairs, transform=None):
        from src.data.splits import image_path
        self.transform = transform
        self.items = [
            (image_path(root_dir, n1, k1), image_path(root_dir, n2, k2), float(label))
            for n1, k1, n2, k2, label in pairs
        ]
        missing = [p for it in self.items for p in it[:2] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} pair images missing, e.g. {missing[:2]}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p1, p2, label = self.items[idx]
        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)


class LFWDataset(Dataset):
    """Random on-the-fly pairs over a set of people. Use for TRAINING only:
    pairs are re-drawn on every access, so it is not a fixed evaluation set."""

    def __init__(self, root_dir, transform=None, split='train', train_ratio=0.8, cache_images=False, people=None):
        """
        LFW Dataset for Siamese Network.
        
        Args:
            root_dir (str): Path to the LFW dataset (e.g., 'data/lfw').
            transform (callable, optional): Optional transform to be applied on a sample.
            split (str): 'train' or 'test'.
            train_ratio (float): Ratio of people to use for training.
            cache_images (bool): If True, loads all images into RAM for faster training.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.cache_images = cache_images
        
        if people is not None:
            # Preferred: explicit identity list from src.data.splits (train_people.txt)
            self.people = sorted(people)
        else:
            # LEGACY 80/20 split (kept so old commands still run). Note it reseeds
            # the GLOBAL `random` module as a side effect.
            all_people = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            all_people.sort() # Ensure deterministic order
            random.seed(42)
            random.shuffle(all_people)
            split_idx = int(len(all_people) * train_ratio)
            if split == 'train':
                self.people = all_people[:split_idx]
            else:
                self.people = all_people[split_idx:]
            
        print(f"[{split.upper()}] Dataset initialized with {len(self.people)} people.")
        
        # Pre-scan images for the selected people
        self.person_images = {}
        self.all_image_paths = []
        
        for person in self.people:
            person_dir = os.path.join(root_dir, person)
            # Support jpg, png, jpeg
            images = glob.glob(os.path.join(person_dir, "*.jpg")) + \
                     glob.glob(os.path.join(person_dir, "*.png")) + \
                     glob.glob(os.path.join(person_dir, "*.jpeg"))
            
            if len(images) > 0:
                self.person_images[person] = images
                self.all_image_paths.extend(images)
                
        # Remove people with 0 images (just in case)
        self.people = [p for p in self.people if p in self.person_images]
        self.people_with_multiple_images = [p for p in self.people if len(self.person_images[p]) > 1]
        
        print(f"[{split.upper()}] Dataset initialized with {len(self.people)} people.")
        print(f"[{split.upper()}] People with >1 images: {len(self.people_with_multiple_images)}")

        self.images_cache = {}
        if self.cache_images:
            print(f"[{split.upper()}] Caching {len(self.all_image_paths)} images to RAM...")
            for img_path in tqdm(self.all_image_paths):
                try:
                    # We open and keep as PIL Image to allow transforms to work seamlessly
                    img = Image.open(img_path).convert("RGB")
                    # Optionally resize here to save RAM if input size is small, 
                    # but we leave it to transform in __getitem__ to be safe with augmentations.
                    # Actually, if we resize here to a slightly larger size (e.g. 256) it would be safer,
                    # but let's just cache raw validation.
                    
                    # Force loading data by converting to numpy and back or just .load()
                    img.load() 
                    self.images_cache[img_path] = img
                except Exception as e:
                    print(f"Could not cache {img_path}: {e}")
            print(f"[{split.upper()}] Caching Complete.")

        # For the dataset length, we can define it arbitrarily or based on epochs.
        # Let's define it as a fixed number or just len(all_image_paths) to approximate an epoch.
        # However, since we generate pairs on the fly, 'len' is virtual.
        # Let's map it to the number of images to have a reasonable epoch size.
        self.dataset_len = len(self.all_image_paths) 

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        """
        Generates a pair of images (Anchor, Other) and a Label (0 or 1).
        50% chance for same person (1), 50% for different person (0).
        """
        # Randomly choose if we want a positive or negative pair
        should_get_same_class = random.randint(0, 1)
        
        if should_get_same_class:
            # POSITIVE PAIR: Same person
            # Must pick a person who HAS multiple images to avoid infinite loops or fallbacks
            if len(self.people_with_multiple_images) == 0:
                 # Fallback if no one has multiple images (unlikely in LFW but possible in tiny subsets)
                 # Force negative
                 should_get_same_class = 0
            else:
                anchor_person = random.choice(self.people_with_multiple_images)
                anchor_img_path = random.choice(self.person_images[anchor_person])
                
                # Same person, different image
                other_person = anchor_person
                other_img_path = random.choice(self.person_images[anchor_person])
                
                # Ensure it's not the exact same file
                # Since we know len >= 2, this loop will terminate
                while other_img_path == anchor_img_path:
                    other_img_path = random.choice(self.person_images[anchor_person])
                label = 1.0

        if not should_get_same_class:
            # NEGATIVE PAIR: Different person
            # Reselect anchor from ALL people (single or multiple images ok)
            anchor_person = random.choice(self.people)
            anchor_img_path = random.choice(self.person_images[anchor_person])
            
            other_person = random.choice(self.people)
            while other_person == anchor_person:
                other_person = random.choice(self.people)
            other_img_path = random.choice(self.person_images[other_person])
            label = 0.0
            
        # Load Images
        try:
            if self.cache_images and anchor_img_path in self.images_cache:
                # Copy to avoid inplace modifications by some transforms if any
                img1 = self.images_cache[anchor_img_path].copy()
            else:
                img1 = Image.open(anchor_img_path).convert("RGB")

            if self.cache_images and other_img_path in self.images_cache:
                img2 = self.images_cache[other_img_path].copy()
            else:
                img2 = Image.open(other_img_path).convert("RGB")
        except Exception as e:
            # In case of file error, just try another index (recursion)
            # Or return the next one
            print(f"Error loading {anchor_img_path} or {other_img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, torch.tensor(label, dtype=torch.float32)
