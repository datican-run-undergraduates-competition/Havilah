import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import Trainer
from engine import Detr
from data import KidneyStoneDataset

import io
import re
import os
import cv2
import base64
import random
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path


!pip install hf_xet
from transformers import DetrImageProcessor, DetrForObjectDetection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"running on {device}")


from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

# file paths
train_folder = Path(r"/kaggle/input/kidney-stone-images/train")
test_folder = Path(r"/kaggle/input/kidney-stone-images/test")
valid_folder = Path(r"/kaggle/input/kidney-stone-images/valid")

train_images = os.listdir(train_folder / "images")
train_labels = os.listdir(train_folder / "labels")

## login to hugging face
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("DATICAN")
login(token=secret_value_0)

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# transforms for the inputs
transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float)
])

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = processor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch


BATCH_SIZE = 16
train_dataset = KidneyStoneDataset(train_folder / "images", train_folder / "labels", transform, processor=processor)
val_dataset = KidneyStoneDataset(valid_folder / "images", valid_folder / "labels", transform, processor=processor)

train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=3)
val_dataloader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=3)


model = Detr(lr=lr, lr_backbone=lr_backbone, weight_decay=weight_decay).to(device)
trainer = Trainer(max_steps=300, gradient_clip_val=0.1)
trainer.fit(model) 

# save trained model to huggingface hub
model.model.push_to_hub("bamswastaken/datican-detr-v2")

