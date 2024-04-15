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


class StringPathStem:
    """
    从路径中解析出 stem
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"multiline": False, "default": ""}),
            },
        }
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("stem", )
    FUNCTION = "stem"
    CATEGORY = "ruiqu/string"

    def stem(self, path):
        return (Path(path).stem, )


class StringAsAny:
    """
    支持匹配任意类型的字符串
    """
    RETURN_TYPES = (any_type,)
    FUNCTION = "get"
    CATEGORY = "ruiqu/any"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string": ("STRING", {"default": "", "multiline": True})}}

    def get(self, string):
        return (string,)


class StringConcatBase:
    """
    字符串的连接
    """
    FUNCTION = "concat"
    CATEGORY = "ruiqu/string"

    def concat(self, prefix, suffix, join_with_space, **kwargs):
        keys = self.__class__.INPUT_TYPES()['optional'].keys()
        keys = sorted(keys)
        result = []
        for key in keys:
            if join_with_space:
                result.append(
                    prefix.strip() + " " + kwargs[key].strip() + " " + suffix.strip()
                )
            else:
                result.append(
                    prefix + kwargs[key] + suffix
                )
        return tuple(result)


class StringConcat1(StringConcatBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"multiline": False, "default": ""}),
                "suffix": ("STRING", {"multiline": False, "default": ""}),
                "join_with_space": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                f"text_{idx}": ("STRING", {"multiline": False, "default": ""})
                for idx in range(1, 2)
            }
        }
    RETURN_TYPES = tuple(("STRING" for idx in range(1, 2)))
    RETURN_NAMES = tuple((f"text_{idx}" for idx in range(1, 2)))


class StringConcat3(StringConcatBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"multiline": False, "default": ""}),
                "suffix": ("STRING", {"multiline": False, "default": ""}),
                "join_with_space": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                f"text_{idx}": ("STRING", {"multiline": False, "default": ""})
                for idx in range(1, 4)
            }
        }
    RETURN_TYPES = tuple(("STRING" for idx in range(1, 4)))
    RETURN_NAMES = tuple((f"text_{idx}" for idx in range(1, 4)))


class StringConcat6(StringConcatBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"multiline": False, "default": ""}),
                "suffix": ("STRING", {"multiline": False, "default": ""}),
                "join_with_space": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                f"text_{idx}": ("STRING", {"multiline": False, "default": ""})
                for idx in range(1, 7)
            }
        }
    RETURN_TYPES = tuple(("STRING" for idx in range(1, 7)))
    RETURN_NAMES = tuple((f"text_{idx}" for idx in range(1, 7)))


class StringConcat9(StringConcatBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"multiline": False, "default": ""}),
                "suffix": ("STRING", {"multiline": False, "default": ""}),
                "join_with_space": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                f"text_{idx}": ("STRING", {"multiline": False, "default": ""})
                for idx in range(1, 10)
            }
        }
    RETURN_TYPES = tuple(("STRING" for idx in range(1, 10)))
    RETURN_NAMES = tuple((f"text_{idx}" for idx in range(1, 10)))
