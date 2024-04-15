"""
自定义的结点
"""

from .nodes_Any import *
from .nodes_Path import *
from .nodes_Text import *
from .nodes_Image import *

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "TermsToList": TermsToList,
    "EvaluateMultiple1": EvaluateMultiple1,
    "EvaluateMultiple3": EvaluateMultiple3,
    "EvaluateMultiple6": EvaluateMultiple6,
    "EvaluateMultiple9": EvaluateMultiple9,
    "EvaluateListMultiple1": EvaluateListMultiple1,
    "EvaluateListMultiple3": EvaluateListMultiple3,
    "EvaluateListMultiple6": EvaluateListMultiple6,
    "EvaluateListMultiple9": EvaluateListMultiple9,
    "ListPath": ListPath,
    "StringPathStem": StringPathStem,
    "StringAsAny": StringAsAny,
    "StringConcat1": StringConcat1,
    "StringConcat3": StringConcat3,
    "StringConcat6": StringConcat6,
    "StringConcat9": StringConcat9,
    "RangeSplit": RangeSplit,
    "ImageDilate": ImageDilate,
    "ImageErode": ImageErode,
    "MaskDilate": MaskDilate,
    "MaskErode": MaskErode,
    "SaveMask": SaveMask,
    "PreviewMask": PreviewMask,
    "VAEDecodeSave": VAEDecodeSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TermsToList": "TermsToList",
    "EvaluateMultiple1": "EvaluateMultiple1",
    "EvaluateMultiple3": "EvaluateMultiple3",
    "EvaluateMultiple6": "EvaluateMultiple6",
    "EvaluateMultiple9": "EvaluateMultiple9",
    "EvaluateListMultiple1": "EvaluateListMultiple1",
    "EvaluateListMultiple3": "EvaluateListMultiple3",
    "EvaluateListMultiple6": "EvaluateListMultiple6",
    "EvaluateListMultiple9": "EvaluateListMultiple9",
    "ListPath": "ListPath",
    "StringPathStem": "String Get Stem From Path",
    "StringAsAny": "String As Any",
    "StringConcat1": "String Concate 1 Parallelly",
    "StringConcat3": "String Concate 3 Parallelly",
    "StringConcat6": "String Concate 6 Parallelly",
    "StringConcat9": "String Concate 9 Parallelly",
    "RangeSplit": "Split Range To Multiple Intervals",
    "ImageDilate": "ImageDilate(膨胀)",
    "ImageErode": "ImageErode(腐蚀)",
    "MaskDilate": "MaskDilate(膨胀)",
    "MaskErode": "MaskErode(腐蚀)",
    "SaveMask": "Save Mask",
    "PreviewMask": "Preview Mask",
    "VAEDecodeSave": "VAE decode and save image to file",
}
