"""
自定义的结点
"""
import os
import sys
import json
import hashlib
import traceback
import math
import time
import random
import torch
from pathlib import Path
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np
import safetensors.torch
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths
import latent_preview

from nodes import SaveImage


class RangeSplit:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"total_begin": ("INT", {"default":  0, "min": -9999, "max": 9999}),
                             "total_end":   ("INT", {"default":  1, "min": -9999, "max": 9999}),
                             "batch":       ("INT", {"default": 16, "min":     1, "max": 9999}),
                             "overlap":     ("INT", {"default":  0, "min":     0, "max": 9999}),
                             "intervals":   ("INT", {"default": -1, "min":    -1, "max": 9999}),  # 保留的区间数量，负数表示全选
                            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", )
    RETURN_NAMES = ("begin", "end", "length", )  # 区间形式：[begin, end), length = end - begin
    OUTPUT_IS_LIST = (True, True, True, )
    FUNCTION = "make_intervals"
    CATEGORY = "ruiqu/range"
    def make_intervals(self, total_begin, total_end, batch, overlap, intervals=-1):
        assert total_begin <= total_end
        assert 0 <= overlap < batch
        begins, ends = list(), list()
        while not (ends and ends[-1] >= total_end):
            if ends:
                begins.append(ends[-1] - overlap)
            else:
                begins.append(total_begin)
            if begins[-1] + batch >= total_end:
                ends.append(total_end)
            else:
                ends.append(begins[-1] + batch)
            # print(begins, ends)
        assert ends[-1] == total_end
        lengths = [
            end - begin for begin, end in zip(begins, ends)
        ]
        if intervals < 0:
            intervals = len(lengths)
        return (begins[:intervals], ends[:intervals], lengths[:intervals], )


def _chunk_tensor(tensor):
    return torch.chunk(tensor, chunks=len(tensor), dim=0)


def _cat_tensor(chunks):
    return torch.cat(chunks, dim=0)


class ImageDilate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "image": ("IMAGE", ),
                        "kernel_height": ("FLOAT", {"default": 5E-3, "min": 0.0, "max": 1.0, "step": 1E-4}),
                        "kernel_width": ("FLOAT", {"default": 5E-3, "min": 0.0, "max": 1.0, "step": 1E-4}),
                     }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dilate"
    CATEGORY = "ruiqu/image"

    def dilate(self, image, kernel_height=5E-3, kernel_width=5E-3):
        _, h, w, c = image.shape
        kh = max(math.ceil(kernel_height * h), 1)
        kw = max(math.ceil(kernel_width * w), 1)
        chunks = _chunk_tensor(image)
        chunks = [
            img_to_tensor(
                Image.fromarray(np.uint8(
                    cv2.dilate(tensor_to_np(image), np.ones((kh, kw), np.uint8), iterations=1)
                ))
            ) for image in chunks
        ]
        out_image = _cat_tensor(chunks)
        return (out_image, )


class ImageErode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "image": ("IMAGE", ),
                        "kernel_height": ("FLOAT", {"default": 5E-3, "min": 0.0, "max": 1.0, "step": 1E-4}),
                        "kernel_width": ("FLOAT", {"default": 5E-3, "min": 0.0, "max": 1.0, "step": 1E-4}),
                     }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "erode"
    CATEGORY = "ruiqu/image"

    def erode(self, image, kernel_height=5E-3, kernel_width=5E-3):
        _, h, w, c = image.shape
        kh = max(math.ceil(kernel_height * h), 1)
        kw = max(math.ceil(kernel_width * w), 1)
        chunks = _chunk_tensor(image)
        chunks = [
            img_to_tensor(
                Image.fromarray(np.uint8(
                    cv2.erode(tensor_to_np(image), np.ones((kh, kw), np.uint8), iterations=1)
                ))
            ) for image in chunks
        ]
        out_image = _cat_tensor(chunks)
        return (out_image, )


class MaskDilate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "mask": ("MASK", ),
                        "kernel_height": ("FLOAT", {"default": 5E-3, "min": 0.0, "max": 1.0, "step": 1E-4}),
                        "kernel_width": ("FLOAT", {"default": 5E-3, "min": 0.0, "max": 1.0, "step": 1E-4}),
                     }
                }
    RETURN_TYPES = ("MASK",)
    FUNCTION = "dilate"
    CATEGORY = "ruiqu/mask"

    def dilate(self, mask, kernel_height=5E-3, kernel_width=5E-3):
        _, h, w = mask.shape
        kh = max(math.ceil(kernel_height * h), 1)
        kw = max(math.ceil(kernel_width * w), 1)
        chunks = _chunk_tensor(mask)
        chunks = [
            img_to_mask(Image.fromarray(np.uint8(
                cv2.dilate(tensor_to_np(mask), np.ones((kh, kw), np.uint8), iterations=1)
            ))) for mask in chunks
        ]
        out_mask = _cat_tensor(chunks)
        return (out_mask, )


class MaskErode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "mask": ("MASK", ),
                        "kernel_height": ("FLOAT", {"default": 5E-3, "min": 0.0, "max": 1.0, "step": 1E-4}),
                        "kernel_width": ("FLOAT", {"default": 5E-3, "min": 0.0, "max": 1.0, "step": 1E-4}),
                     }
                }
    RETURN_TYPES = ("MASK",)
    FUNCTION = "erode"
    CATEGORY = "ruiqu/mask"

    def erode(self, mask, kernel_height=5E-3, kernel_width=5E-3):
        _, h, w = mask.shape
        kh = max(math.ceil(kernel_height * h), 1)
        kw = max(math.ceil(kernel_width * w), 1)
        chunks = _chunk_tensor(mask)
        chunks = [
            img_to_mask(Image.fromarray(np.uint8(
                cv2.erode(tensor_to_np(mask), np.ones((kh, kw), np.uint8), iterations=1)
            ))) for mask in chunks
        ]
        out_mask = _cat_tensor(chunks)
        return (out_mask, )


class SaveMask(SaveImage):
    """
    保存掩码
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"masks": ("MASK", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI_Mask"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
    RETURN_TYPES = ()
    FUNCTION = "save_masks"
    OUTPUT_NODE = True
    CATEGORY = "ruiqu/mask"

    def save_masks(self, masks, filename_prefix="ComfyUI_Mask", prompt=None, extra_pnginfo=None):
        images = masks.reshape((-1, 1, masks.shape[-2], masks.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return self.save_images(images, filename_prefix=filename_prefix, prompt=prompt, extra_pnginfo=extra_pnginfo)


class PreviewMask(SaveMask):
    """
    预览掩码
    """
    def __init__(self):
        super().__init__()
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return SaveMask.INPUT_TYPES()


class VAEDecodeSave:
    """
    对潜空间采样进行解码并保存为图像文件
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "samples": ("LATENT", ),
                    "vae": ("VAE", ),
                    "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    RETURN_TYPES = ()
    FUNCTION = "decode_save"
    OUTPUT_NODE = True
    CATEGORY = "ruiqu/latent"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    def decode_save(self, vae, samples, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        images = vae.decode(samples["samples"])
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}


def img_to_tensor(input):
    i = ImageOps.exif_transpose(input)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image)[None,]
    return tensor


def img_to_np(input):
    i = ImageOps.exif_transpose(input)
    image = i.convert("RGB")
    image_np = np.array(image).astype(np.float32)
    return image_np


def img_to_mask(input):
    i = ImageOps.exif_transpose(input)
    image = i.convert("RGB")
    new_np = np.array(image).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(new_np).permute(2, 0, 1)[0:1, :, :]
    return mask_tensor


def np_to_tensor(input):
    image = input.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image)[None,]
    return tensor


def tensor_to_img(image):
    image = image[0]
    i = 255. * image.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB")
    return img


def tensor_to_np(image):
    image = image[0]
    i = 255. * image.cpu().numpy()
    result = np.clip(i, 0, 255).astype(np.uint8)
    return result


def np_to_mask(input):
    new_np = input.astype(np.float32) / 255.0
    tensor = torch.from_numpy(new_np).permute(2, 0, 1)[0:1, :, :]
    return tensor
