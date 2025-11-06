import torch
import math
import inspect
import logging
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, List
from transformers import LlavaConfig, PretrainedConfig, PreTrainedModel

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def find_all_linear_modules(model: "PreTrainedModel", forbidden_modules: dict) -> List[str]:
    r"""
    Finds all available modules to apply lora or galore.
    """
    model_type = getattr(model.config, "model_type", None)
    if len(forbidden_modules) == 0:
        forbidden_modules.add("lm_head")
        if model_type == "chatglm":
            forbidden_modules.add("output_layer")
        elif model_type == "internlm2":
            forbidden_modules.add("output")
        elif model_type in ["llava", "paligemma"]:
            forbidden_modules.add("multi_modal_projector")
        elif model_type == "qwen2_vl":
            forbidden_modules.add("merger")
            forbidden_modules.add("visual")

    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])
    
    # logger.info("Found linear modules: {}".format(",".join(module_names)))
    print("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    else:
        to_return = {(k[17:] if k.startswith('base_model.model.') else k): t for k, t in to_return.items()}
    to_return = {(k.replace('modules_to_save.default.','') if 'modules_to_save' in k else k): t for k, t in to_return.items()}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return