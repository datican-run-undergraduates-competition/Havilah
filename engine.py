"""
    The Model to be trained
"""

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset, DataLoader

import numpy as np

import pytorch_lightning as pl
from transformers import DetrImageProcessor, DetrForObjectDetection

class Detr(pl.LightningModule):
     def __init__(self, lr, lr_backbone, weight_decay):
         super().__init__()
         # replace COCO classification head with custom head
         # we specify the "no_timm" variant here to not rely on the timm library
         # for the convolutional backbone
         self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                             revision="no_timm",
                                                             num_labels=1,
                                                             ignore_mismatched_sizes=True)
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay

     def forward(self, pixel_values, pixel_mask,output_attentions=True):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, output_attentions=output_attentions)
       return outputs

     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        
        # Get actual batch size (number of samples in this batch)
        batch_size = batch["pixel_values"].shape[0]  
        
        # Log with explicit batch size
        self.log("training_loss", loss, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item(), batch_size=batch_size)
        
        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        
        batch_size = batch["pixel_values"].shape[0]
        
        self.log("validation_loss", loss, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"validation_{k}", v.item(), batch_size=batch_size)
        
        return loss

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]

        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

        return optimizer

     def train_dataloader(self):
        return train_dataloader

     def val_dataloader(self):
        return val_dataloader
