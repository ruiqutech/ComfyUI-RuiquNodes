"""
自定义的结点
"""
import os
import sys
import copy
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


# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


# Our any instance wants to be a wildcard string
any_type = AnyType("*")


def show_any(obj, max_len):
    if isinstance(obj, torch.Tensor):
        result = f'shape {tuple(obj.shape)}, min {obj.min()}, max {obj.max()}, median {torch.median(obj)}, content: {str(obj)}'
    elif isinstance(obj, numpy.ndarray):
        result = f'shape {tuple(obj.shape)}, min {obj.min()}, max {obj.max()}, median {numpy.median(obj)}, content: {str(obj)}'
    elif isinstance(obj, list) or isinstance(obj, tuple):
        result = 'list' if isinstance(obj, list) else 'tuple'
        result += f'({len(obj)})'
        if obj:
            result += f', obj[0]: {show_any(obj[0], max_len=max_len / len(obj))}'
    else:
        result = str(obj)
    if max_len > 0:
        result = result[:max_len]
    return result


def _print_to_console(header='', max_len=0, **kwargs):
    if header:
        print(f"\033[01;33m{header}\033[0m")
    for key, value in kwargs.items():
        print(f"\033[01;33m{key} = {show_any(value, max_len)}\n\033[0m")


def _exec(expression, output_names, **kwargs):
    try:
        local_vars = dict()
        exec(expression, kwargs.copy(), local_vars)
        result = {
            name: local_vars.pop(name, None) for name in output_names
        }
        return result
    except Exception as ex:
        _print_to_console(header=f"Error expression:\n\n{expression}\n\noutput_names = {output_names}\nex = {ex}")
        raise ex


def _evaluate(self, input_names, output_names, **kwargs):
    # 找到没有经过修改的 expression
    unique_id = int(kwargs.pop('unique_id'))
    workflow = kwargs.pop('extra_pnginfo')['workflow']
    nodes = workflow['nodes']
    found = False
    for one_node in nodes:
        if one_node['id'] == unique_id:
            assert not found
            found = True
            index = list(type(self).INPUT_TYPES()['required'].keys()).index('expression')
            expression = one_node['widgets_values'][index]
            kwargs['expression'] = expression
    assert found
    # 计算该 expression，并获得结果
    input = {
        name: kwargs.pop(name, None) for name in input_names
    }
    output = _exec(expression=expression, output_names=output_names, **input)
    if kwargs.pop('print_to_console') == "True":
        _print_to_console(header='Evaluate', expression=expression, **input, **output)
    return tuple((
        output[name] for name in output_names
    ))


class EvaluateMultiple1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "", "multiline": True}),
                "input_count": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "print_to_console": (["False", "True"],),
            },
            "optional": {
                f'in{idx}': (any_type, {}) for idx in range(1, 2)
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    RETURN_TYPES = tuple([any_type] * 1)
    RETURN_NAMES = tuple([f'out{idx}' for idx in range(1, 2)])
    FUNCTION = "evaluate"
    CATEGORY = "ruiqu/evaluate"

    def evaluate(self, **kwargs):
        input_names = sorted(type(self).INPUT_TYPES()['optional'].keys())
        output_names = sorted(type(self).RETURN_NAMES)
        return _evaluate(self, input_names=input_names, output_names=output_names, **kwargs)


class EvaluateMultiple3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "", "multiline": True}),
                "input_count": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "print_to_console": (["False", "True"],),
            },
            "optional": {
                f'in{idx}': (any_type, {}) for idx in range(1, 4)
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    RETURN_TYPES = tuple([any_type] * 3)
    RETURN_NAMES = tuple([f'out{idx}' for idx in range(1, 4)])
    FUNCTION = "evaluate"
    CATEGORY = "ruiqu/evaluate"

    def evaluate(self, **kwargs):
        input_names = sorted(type(self).INPUT_TYPES()['optional'].keys())
        output_names = sorted(type(self).RETURN_NAMES)
        return _evaluate(self, input_names=input_names, output_names=output_names, **kwargs)


class EvaluateMultiple6:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "", "multiline": True}),
                "input_count": ("INT", {"default": 6, "min": 1, "max": 20, "step": 1}),
                "print_to_console": (["False", "True"],),
            },
            "optional": {
                f'in{idx}': (any_type, {}) for idx in range(1, 7)
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    RETURN_TYPES = tuple([any_type] * 6)
    RETURN_NAMES = tuple([f'out{idx}' for idx in range(1, 7)])
    FUNCTION = "evaluate"
    CATEGORY = "ruiqu/evaluate"

    def evaluate(self, **kwargs):
        input_names = sorted(type(self).INPUT_TYPES()['optional'].keys())
        output_names = sorted(type(self).RETURN_NAMES)
        return _evaluate(self, input_names=input_names, output_names=output_names, **kwargs)


class EvaluateMultiple9:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "", "multiline": True}),
                "input_count": ("INT", {"default": 9, "min": 1, "max": 20, "step": 1}),
                "print_to_console": (["False", "True"],),
            },
            "optional": {
                f'in{idx}': (any_type, {}) for idx in range(1, 10)
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    RETURN_TYPES = tuple([any_type] * 9)
    RETURN_NAMES = tuple([f'out{idx}' for idx in range(1, 10)])
    FUNCTION = "evaluate"
    CATEGORY = "ruiqu/evaluate"

    def evaluate(self, **kwargs):
        input_names = sorted(type(self).INPUT_TYPES()['optional'].keys())
        output_names = sorted(type(self).RETURN_NAMES)
        return _evaluate(self, input_names=input_names, output_names=output_names, **kwargs)


class EvaluateListMultipleBase:
    FUNCTION = "evaluate"
    CATEGORY = "ruiqu/evaluate"

    def evaluate(self, **kwargs):
        # 输入
        input_names = sorted(type(self).INPUT_TYPES()['optional'].keys())
        intput_length = max([len(kwargs[name]) for name in input_names if name in kwargs])
        assert all([intput_length == len(kwargs[name]) for name in input_names if name in kwargs])
        # 输出
        output_names = sorted(type(self).RETURN_NAMES)
        output_values = [list() for _ in range(len(output_names))]  # 不能使用 * len(output_names)，会拷贝地址
        # 计算
        for index in range(intput_length):
            kwargs_term = copy.deepcopy(kwargs)
            for name in set(input_names) & set(kwargs_term):
                kwargs_term[name] = [kwargs_term[name][index]]  # 提取并统一形式
            for name in kwargs_term:
                assert kwargs_term[name]
                kwargs_term[name] = kwargs_term[name][0]  # 提取列表元素
            values = _evaluate(self, input_names=input_names, output_names=output_names, **kwargs_term)
            assert len(values) == len(output_values)
            if all([_ is None for _ in values]):
                continue
            for idx in range(len(values)):
                output_values[idx].append(values[idx])
        return (tuple(output_values), )


class EvaluateListMultiple1(EvaluateListMultipleBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "# 表达式为element-wise，合并时自动跳过全None的输出", "multiline": True}),
                "input_count": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "print_to_console": (["False", "True"],),
            },
            "optional": {
                f'in{idx}': (any_type, {}) for idx in range(1, 2)
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = tuple([True] * 1)
    RETURN_TYPES = tuple([any_type] * 1)
    RETURN_NAMES = tuple([f'out{idx}' for idx in range(1, 2)])


class EvaluateListMultiple3(EvaluateListMultipleBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "# 表达式为element-wise，合并时自动跳过全None的输出", "multiline": True}),
                "input_count": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "print_to_console": (["False", "True"],),
            },
            "optional": {
                f'in{idx}': (any_type, {}) for idx in range(1, 4)
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = tuple([True] * 3)
    RETURN_TYPES = tuple([any_type] * 3)
    RETURN_NAMES = tuple([f'out{idx}' for idx in range(1, 4)])


class EvaluateListMultiple6(EvaluateListMultipleBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "# 表达式为element-wise，合并时自动跳过全None的输出", "multiline": True}),
                "input_count": ("INT", {"default": 6, "min": 1, "max": 20, "step": 1}),
                "print_to_console": (["False", "True"],),
            },
            "optional": {
                f'in{idx}': (any_type, {}) for idx in range(1, 7)
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = tuple([True] * 6)
    RETURN_TYPES = tuple([any_type] * 6)
    RETURN_NAMES = tuple([f'out{idx}' for idx in range(1, 7)])


class EvaluateListMultiple9(EvaluateListMultipleBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "# 表达式为element-wise，合并时自动跳过全None的输出", "multiline": True}),
                "input_count": ("INT", {"default": 9, "min": 1, "max": 20, "step": 1}),
                "print_to_console": (["False", "True"],),
            },
            "optional": {
                f'in{idx}': (any_type, {}) for idx in range(1, 10)
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = tuple([True] * 9)
    RETURN_TYPES = tuple([any_type] * 9)
    RETURN_NAMES = tuple([f'out{idx}' for idx in range(1, 10)])


class TermsToList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "terms": (any_type, {}),
            }
        }
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = (any_type, )
    RETURN_NAMES = ("one_list", )
    FUNCTION = "convert"
    CATEGORY = "ruiqu/list"
    def convert(self, terms):
        return(terms, )
