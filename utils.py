import io 
import os
import cv2
import base64
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from transformers import DetrImageProcessor, DetrForObjectDetection

def get_attention_map(postprocessed_outputs, outputs):
    """
    Gets attention map from bounding box with highest confidence.
    Args:
        postprocessed_outputs: [scores, labels, boxes]
        outputs: model output
    Returns:
        2d attention map
    """
    scores = postprocessed_outputs[0]['scores']
    if len(scores) > 0:
        highest_conf_idx = scores.argmax()
        highest_conf_query_idx = highest_conf_idx.item()  # This is the query index for the highest confidence box
    highest_conf_query_idx

    layer_idx = -1 # last layer, usually more refined
    head_idx = 6

    cross_attentions = outputs.cross_attentions
    attn_weights = cross_attentions[layer_idx][0]
    # attention_map = attn_weights[head_idx, highest_conf_query_idx] # attention map for bbox with largest confidence
    attention_map = attn_weights.mean(dim=0)[highest_conf_query_idx] # attention map for bbox with largest confidence averaged over all heads


     # Validate attention map size
    num_elements = attention_map.numel()
    if num_elements == 775: 
        attention_map_2d = attention_map.reshape(25, int(775 / 25))
    else:
        # Fallback for unexpected sizes
        side_length = int(num_elements**0.5)
        print(f"error_zone: {attention_map.shape}")
        attention_map_2d = attention_map.reshape(25, -1) # infer dim


    # reshape attention map
    # attention_map_2d = attention_map.reshape(25, int(775 / 25)) # this causes an error

    return attention_map_2d

def attention_to_overlay(original_image, attention_map, plot=False):
    """
    Overlays attention heatmap on the original image.
    Args:
        original_image: PIL Image (RGB)
        attention_map: 2D torch tensor (H, W)
        plot: plot image
    Returns:
        PIL Image (RGB) with heatmap overlay in base64 format
    """
    # Convert original image to numpy array
    original = np.array(original_image)
    
    # Resize attention map to match original image dimensions
    attention_resized = F.interpolate(
        attention_map.unsqueeze(0).unsqueeze(0).float(),
        size=(original.shape[0], original.shape[1]),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()
    
    # Normalize attention map to [0, 1]
    attention_normalized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
    
    # Apply colormap (viridis) to attention map
    heatmap = plt.cm.viridis(attention_normalized)[..., :3]  # RGBA → RGB, range [0, 1]
    
    # Blend original image and heatmap (50% opacity)
    overlayed = 0.5 * original + 0.5 * heatmap * 255
    overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)
    
    # converts numpy array to PIL Image
    
    if plot:
        plt.imshow(overlayed)
        plt.title("Cross Attention Map of Bounding box with the highest accuracy")
        plt.axis('off')
        plt.show()
    else:    
        attn_img_overlay = Image.fromarray(overlayed) 
    
        buffered = io.BytesIO()
        attn_img_overlay.save(buffered, format="PNG")
        atten_img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return atten_img_str 

def scale_bbox_to_img_size(preprocessed_img, image, boxes):
    """
    Scales bounding box predictions to actual image size
    Args:
        preprocessed_img: image passed through huggingface transforms
        image: PIL image
        boxes: bounding box predictions
    Returns:
        scaled bounding boxes
    """
    pp_width, pp_height = preprocessed_img.shape[-2:] # (1,3,800,977)
    image_height, image_width = image.size
    
    # bbox is in pp_image format, scale to actual image format
    width_scale = image_width / pp_width
    height_scale =  image_height / pp_height

    new_boxes = []
    for box in boxes:

        if torch.is_tensor(box):
            box = box.cpu().numpy()

        scaled_box = np.array(box, dtype=np.float32)

        scaled_box[0:3:2] = scaled_box[0:3:2] * width_scale # the width (x1, x2)
        scaled_box[1:4:2] = scaled_box[1:4:2] * height_scale # the height (y1, y2)
        new_boxes.append(scaled_box)
        
    return new_boxes

def image_with_bboxes(pil_img, scores, labels, scaled_boxes, plot=False):
    """
    Scales bounding box predictions to actual image size
    Args:
        pil_img: image 
        scores: detection confidences
        scaled_boxes: scaled bounding box predictions
        plot: plot image
    Returns:
        PIL Image (RGB) with bounding boxes in base64 format
    """
    
    img_np = np.array(pil_img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    boxcolor = (50,200,50)
    textcolor = (255,255,255)

    for score, label, (xmin, ymin, xmax, ymax) in zip(scores, labels, scaled_boxes):
        cv2.rectangle(img_cv, (int(xmin), int(ymin)), (int(xmax), int(ymax)), boxcolor, 1)
        cv2.putText(img_cv, f'stone: {score:.2f}', (int(xmin), int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, textcolor)

    if plot:
        plt.imshow(img_cv)
        plt.axis('off')
        plt.show
    else:
        bbox_on_image = Image.fromarray(img_cv)
        buffered = io.BytesIO()
        bbox_on_image.save(buffered, format="PNG")
        bbox_on_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return bbox_on_image_str
