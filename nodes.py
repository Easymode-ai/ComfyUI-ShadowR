from PIL import Image
from torch.utils.data import Dataset
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
from torchvision import transforms
import os
import time
import re
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from .shadow_r.models import final_net

import folder_paths

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar

script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model_dir = os.path.join(comfy_path, "models", "shadow_r")

def process_chunk(model, chunk, device):
    with torch.no_grad():
        chunk = chunk.to(device)
        return model(chunk)

def split_image(img_tensor, window_size=512, overlap=64):
    """Split image into overlapping chunks."""
    _, _, h, w = img_tensor.shape
    chunks = []
    positions = []
    
    for y in range(0, h, window_size - overlap):
        for x in range(0, w, window_size - overlap):
            # Calculate chunk boundaries
            y1 = y
            x1 = x
            y2 = min(y + window_size, h)
            x2 = min(x + window_size, w)
            
            # Extract chunk
            chunk = img_tensor[:, :, y1:y2, x1:x2]
            
            # Pad if necessary to maintain consistent size
            if chunk.shape[2] < window_size or chunk.shape[3] < window_size:
                ph = window_size - chunk.shape[2]
                pw = window_size - chunk.shape[3]
                chunk = F.pad(chunk, (0, pw, 0, ph))
            
            chunks.append(chunk)
            positions.append((y1, y2, x1, x2))
    
    return chunks, positions
def merge_chunks(chunks, positions, original_size, window_size=512, overlap=64):
    """Merge overlapping chunks with linear blending."""
    h, w = original_size
    result = torch.zeros((1, 3, h, w), device=chunks[0].device)
    weights = torch.zeros((1, 3, h, w), device=chunks[0].device)
    
    for chunk, (y1, y2, x1, x2) in zip(chunks, positions):
        # Calculate actual chunk size (might be smaller at edges)
        chunk_h = y2 - y1
        chunk_w = x2 - x1
        
        # Extract the valid portion of the chunk (without padding)
        valid_chunk = chunk[:, :, :chunk_h, :chunk_w]
        
        # Create weight mask
        weight = torch.ones_like(valid_chunk)
        
        # Apply linear blending in overlap regions
        if overlap > 0:
            for i in range(overlap):
                weight_value = i / overlap
                # Blend left edge if not at image boundary
                if x1 > 0 and i < chunk_w:
                    weight[:, :, :, i] *= weight_value
                # Blend right edge if not at image boundary
                if x2 < w and chunk_w - i - 1 >= 0:
                    weight[:, :, :, -(i + 1)] *= weight_value
                # Blend top edge if not at image boundary
                if y1 > 0 and i < chunk_h:
                    weight[:, :, i, :] *= weight_value
                # Blend bottom edge if not at image boundary
                if y2 < h and chunk_h - i - 1 >= 0:
                    weight[:, :, -(i + 1), :] *= weight_value
        
        result[:, :, y1:y2, x1:x2] += valid_chunk * weight
        weights[:, :, y1:y2, x1:x2] += weight
    
    # Normalize by weights to complete blending
    valid_mask = weights > 0
    result[valid_mask] = result[valid_mask] / weights[valid_mask]
    
    return result

def get_filename_list(folder_name: str):
    files = [f for f in os.listdir(folder_name)]
    return files
    
# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)    
    
class images_dataset(Dataset):
    def __init__(self, images):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_images=[]
        
        input_images = images.permute(0,3,1,2)

        for i in input_images:
            self.list_images.append(i)

        self.nb_images = len(self.list_images)

    def __getitem__(self, index, is_train=True):        
        #image = transforms.functional.to_pil_image(self.list_images[index], mode='RGB')
        #image = Image.open(self.list_images[index]).convert('RGB')
        #image = self.transform(image)
        #image = transforms.functional.to_pil_image(self.list_images[index], mode='RGB')
        #image = self.transform(image)
        image = self.list_images[index]
        
        return image
        
    def __len__(self):
        return self.nb_images

class ShadowRModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shadowremoval_model": (get_filename_list(model_dir), {"tooltip": "These models are loaded from the 'ComfyUI/models/shadow_r' folder",}),
                "enhancement_model": (get_filename_list(model_dir), {"tooltip": "These models are loaded from the 'ComfyUI/models/shadow_r' folder",}),
            }
        }

    RETURN_TYPES = ("SHADOWRMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loadmodel"
    CATEGORY = "ShadowRWrapper"
    
    def loadmodel(self, shadowremoval_model, enhancement_model):
        model = final_net()
        
        shadowremoval_model_path = os.path.join(model_dir, shadowremoval_model)
        enhancement_model_path = os.path.join(model_dir, enhancement_model)
        
        model.remove_model.load_state_dict(torch.load(shadowremoval_model_path, map_location='cpu'), strict=True)
        model.enhancement_model.load_state_dict(torch.load(enhancement_model_path, map_location='cpu'), strict=True)        
        
        device = mm.get_torch_device()
        
        model = model.to(device)
        
        return (model,)
        
class ShadowRShadowRemover:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("SHADOWRMODEL",),
                "image": ("IMAGE", ),
		"chunk_size": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 32}),
		"overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 32})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "removeshadow"
    CATEGORY = "ShadowRWrapper"
    
    def removeshadow(self, model, image, chunk_size, overlap):
        device = mm.get_torch_device()
        dataset = images_dataset(image)
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)       
        
        images_out = []
        
        with torch.no_grad():
            model.eval()

            for batch_idx, (input) in enumerate(loader):
                input = input.to(device)
                
                # Get image dimensions
                _, _, h, w = input.shape
                print(f"Image size: {w}x{h}")

                # Split image into chunks
                chunks, positions = split_image(input, chunk_size, overlap)
                print(f"Split into {len(chunks)} chunks")
            
               # Process each chunk
                processed_chunks = []
                for i, chunk in enumerate(chunks):
                   print(f"Processing chunk {i+1}/{len(chunks)}")
                   processed_chunk = process_chunk(model, chunk, device)
                   processed_chunks.append(processed_chunk)
                   torch.cuda.empty_cache()  # Clear GPU memory after each chunk
            
                # Merge chunks
                image_result = merge_chunks(processed_chunks, positions, (h, w), chunk_size, overlap)


                #image_result = model(input)
                #image_result = image_result.to(device)
                
                # Debugging: Print type and shape
                print(f"Type of image_result: {type(image_result)}")
                print(f"Shape of image_result: {image_result.shape}")
                print(f"Dtype of image_result: {image_result.dtype}")

                # Ensure tensor is in the correct format for ComfyUI
                image_result = image_result.permute(0, 2, 3, 1).cpu().contiguous()  # Convert to (B, H, W, C)
                image_result = image_result.clamp(0, 1).float()  # Ensure values are in [0,1]

                images_out.append(image_result)
        
        #torch_images = torch.cat(images_out, dim=0)
        
        return (image_result,)

NODE_CLASS_MAPPINGS = {
    "ShadowRModelLoader": ShadowRModelLoader,
    "ShadowRShadowRemover": ShadowRShadowRemover
    }
    
NODE_DISPLAY_NAME_MAPPINGS = {
    "ShadowRModelLoader": "ShadowR Model Loader",
    "ShadowRShadowRemover": "ShadowR Shadow Remover"
    }