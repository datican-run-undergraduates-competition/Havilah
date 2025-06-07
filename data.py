"""
Class to handle datsets
"""

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset, DataLoader

import re
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

from transformers import DetrImageProcessor


class KidneyStoneDataset(Dataset):
    """
        Dataset Processor for Kidney stone dataset
    """
    def __init__(self, imgs_path, labels_path, transform, processor):
        super(KidneyStoneDataset, self).__init__()
        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.transform = transform
        self.processor = processor
        self.images, self.annotations_df = self.yolo2coco(imgs_path, labels_path)

        ### deepseek!!
        valid_images = []
        for img in self.images:
            if img in self.annotations_df['images'].values:
                valid_images.append(img)
                self.images = valid_images

    def yolo2coco(self, images_path, labels_path):
        annotations = {
        "images" : [],
        "class_id" : [],
        "x_center_norm" : [],
        "y_center_norm" : [],
        "width_norm" : [],
        "height_norm" : [],
        "width_act" : [],
        "height_act" : []
        }

        images = os.listdir(images_path)
    
        for image in images:
            
            # open label folder
            label_name = image[:-3] + "txt"
            label_path = labels_path / label_name
            img_path = images_path / image
    
            # Add this check
            if not os.path.exists(label_path):
                print(f"Label file missing for {image}, skipping")
                continue
            
            with open(label_path, 'r') as f:
                labels = f.readlines()

            width, height = Image.open(img_path).size # width, height

            if len(labels) == 1:
                annotations["images"].append(image)

                labels = labels[0]
                labels = re.sub(r"\n", "", labels)
                labels = re.sub(r",", "", labels)
                labels = labels.strip()
                id, x_cen, y_cen, w, h = [float(val) for val in labels.split(" ")]
            
                annotations["class_id"].append(id)
                annotations["x_center_norm"].append(x_cen)
                annotations["y_center_norm"].append(y_cen)
                annotations["width_norm"].append(w)
                annotations["height_norm"].append(h)
                annotations["width_act"].append(width)
                annotations["height_act"].append(height)
                
            elif len(labels) > 1:
                for label in labels:
                    label = re.sub(r"\n", " ", label)
                    label = re.sub(r",", " ", label)
                    label = label.strip()
                    id_, x_cen, y_cen, w, h = [float(val) for val in label.split(" ")]
                    annotations["images"].append(image)
                    annotations["class_id"].append(id_)
                    annotations["x_center_norm"].append(x_cen)
                    annotations["y_center_norm"].append(y_cen)
                    annotations["width_norm"].append(w)
                    annotations["height_norm"].append(h)
                    annotations["width_act"].append(width)
                    annotations["height_act"].append(height)
                
            else:
                # remove image from list of image
                img_idx = images.index(image)
                images = images[:img_idx] + images[img_idx:]

                print(f"image {image} has no labels, it has been removed from the dataset")

        annot_df = pd.DataFrame(annotations)


        # denormalize yolo format
        annot_df["x_center_abs"] = annot_df["x_center_norm"] * annot_df['width_act']
        annot_df["y_center_abs"] = annot_df["y_center_norm"] * annot_df["height_act"]
        annot_df["width_abs"] = annot_df["width_norm"] * annot_df['width_act']
        annot_df["height_abs"] = annot_df["height_norm"] * annot_df["height_act"]

        # convert to coco
        annot_df["x_min"] = annot_df["x_center_abs"] - (annot_df["width_abs"] / 2)
        annot_df["y_min"] = annot_df["y_center_abs"] - (annot_df["height_abs"] / 2)

        annot_df["class_id"] = annot_df["class_id"].astype('int')
        annot_df["x_min"] = annot_df["x_min"].apply(lambda x : round(x,1))
        annot_df["y_min"] = annot_df["y_min"].apply(lambda x : round(x,1))
        annot_df["width_abs"] = annot_df["width_abs"].apply(lambda x : round(x,1))
        annot_df["height_abs"] = annot_df["height_abs"].apply(lambda x : round(x,1))

        annot_df = annot_df[['images', 'class_id', 'x_min', 'y_min', 'width_abs', 'height_abs']]

        return images, annot_df 


    def __len__(self):
        return len(self.images)

  
    def __getitem__(self, idx):
        image_name = self.images[idx]
        img_path = self.imgs_path / image_name
        img = Image.open(img_path).convert("RGB")  # Ensure RGB format

        if self.transform:
            img = self.transform(img)
    
        # Get annotations for this image
        image_annots = self.annotations_df[self.annotations_df['images'] == image_name]
        
        # Convert to COCO format
        annotations = []
        for _, row in image_annots.iterrows():
            annotation = {
                "bbox": [
                    row['x_min'],
                    row['y_min'],
                    row['width_abs'],
                    row['height_abs']
                ],
                "category_id": row['class_id'],
                "area": row['width_abs'] * row['height_abs'],
                "iscrowd": 0
            }
            annotations.append(annotation)
        
        # Create target in COCO format
        target = {
            "image_id": torch.tensor([idx]),  # Using index as image_id
            "annotations": annotations
        }
        
        # Process with DETR processor
        encoding = self.processor(
            images=img, 
            annotations=target, 
            return_tensors="pt"
        )
        
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension
    
        return pixel_values, target
