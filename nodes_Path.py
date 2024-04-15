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
import numpy
import torch
import numbers
from pathlib import Path
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np
import safetensors.torch

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

from .nodes_Any import any_type, show_any


def _gen_list_files(dirname):
    """列出目录下所有的文件"""
    for root, dirs, files in os.walk(dirname, topdown=False):
        for name in files:
            yield os.path.join(root, name)


class ListPath:
    """
    从路径中解析出 stem
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "prefix": ("STRING", {"multiline": False, "default": ""}),
                "suffix": ("STRING", {"multiline": False, "default": ""}),
                "delimiter": ("STRING", {"multiline": False, "default": ","}),
                "count": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING", )
    RETURN_NAMES = ("path", "show_help", )
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "make_list"
    CATEGORY = "ruiqu/path"

    def make_list(self, path, prefix, suffix, delimiter, count):
        show_help = "delimiter非空时候，会对prefix和suffix进行split"
        if delimiter:
            prefix = [_.strip() for _ in prefix.split(delimiter) if _.strip()]
            suffix = [_.strip() for _ in suffix.split(delimiter) if _.strip()]
        else:
            prefix = []
            suffix = []
        result = []
        if Path(path).is_file():
            result.append(path)
        elif Path(path).is_dir():
            for aPath in _gen_list_files(path):
                if prefix and not any([Path(aPath).name.startswith(_) for _ in prefix]):
                    continue
                if suffix and not any([Path(aPath).name.endswith(_) for _ in suffix]):
                    continue
                result.append(aPath)
        result.sort()
        if count > 0:
            result = result[:count]
        return (result, show_help)
