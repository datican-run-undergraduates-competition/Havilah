# imports!
import io 
import os
import cv2
import base64
import pickle
import uvicorn
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, Request


import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from transformers import DetrImageProcessor, DetrForObjectDetection

from utils import get_attention_map, attention_to_overlay, scale_bbox_to_img_size, image_with_bboxes

model = DetrForObjectDetection.from_pretrained("bamswastaken/datican-detr-v2", num_labels=1, ignore_mismatched_sizes=True)
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

app = FastAPI()
 
# on load
@app.get('/')
async def ok():
    return {'status code':'200 - she works!'}


@app.post("/datican")
async def datican(request: Request, file: UploadFile = File(...)):
    img_file = await file.read()
    img = Image.open(io.BytesIO(img_file)).convert("RGB")

    # apply transforms
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float)
    ])

    img_trans = transform(img).unsqueeze(dim=0)

    encoding = processor(images=img_trans, return_tensors="pt")

    # input image with transforms applied
    preprocessed_img = encoding['pixel_values'] # b,c,h,w

    # print(f"image shape: {img_trans.shape}")

    # get bboxes and confidences
    with torch.no_grad():
        # forward pass to get class logits and bounding boxes
        outputs = model(pixel_values=preprocessed_img, pixel_mask=None,output_attentions=True)
        # ['logits', 'pred_boxes', 'last_hidden_state', 'decoder_attentions', 'cross_attentions', 'encoder_last_hidden_state', 'encoder_attentions'])


    postprocessed_outputs = processor.post_process_object_detection(outputs,
                                                                    target_sizes=[(preprocessed_img.shape[2], preprocessed_img.shape[3])],
                                                                    threshold=0.6)
    bboxes = postprocessed_outputs[0]

    # make web useable
    bboxes = {k: v.numpy().tolist() for k,v in bboxes.items()} # if on gpu put .detach().cpu()
    print(bboxes)

    boxes = scale_bbox_to_img_size(preprocessed_img, img, bboxes['boxes'])
    img_with_bboxes = image_with_bboxes(img, bboxes['scores'], bboxes['labels'], boxes, plot=False)

    #  use scaled boxes
    bboxes['boxes'] = [box.tolist() for box in boxes]

    # checks if bboxes are empty
    if len(bboxes['scores']) == 0:
        return {"bboxes":bboxes}
    else:
        attn_map = get_attention_map(postprocessed_outputs, outputs)
        atten_img = attention_to_overlay(img, attn_map,plot=False)

        return {"bboxes":bboxes,"img_with_bboxes":img_with_bboxes,"cross_attention_map":atten_img}


# run app 
if __name__ == "__main__":
    uvicorn.run(app)

# http://127.0.0.1:8000
