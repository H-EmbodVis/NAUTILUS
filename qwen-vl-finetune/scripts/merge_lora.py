import argparse
import torch
from torch import nn
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from transformers import AutoTokenizer

from qwenvl.nautilus_model.Qwen2_5_VL_Nautilus_ForConditionalGeneration import Qwen2_5_VL_Nautilus_ForConditionalGeneration
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file

def remap_non_lora_keys(adapter_state_dict, model):
    new_state_dict = {}

    modules_to_save_names = []
    for name, module in model.named_modules():
        if hasattr(module, "modules_to_save") and isinstance(module.modules_to_save, torch.nn.ModuleDict):
            modules_to_save_names.append(name)
    
    for k, v in adapter_state_dict.items():
        if "lora_" in k:
            pass
        else:
            if k.startswith("base_model.model."):
                k = k[len("base_model.model."):]  
            
            inserted = False
            for target_prefix in modules_to_save_names:
                if k.startswith(target_prefix):
                    parts = k.split(".")
                    prefix_len = len(target_prefix.split("."))
                    new_key = ".".join(parts[:prefix_len]) + ".modules_to_save.default." + ".".join(parts[prefix_len:])
                    new_state_dict[new_key] = v
                    inserted = True
                    break
            if not inserted:
                new_state_dict[k] = v

    return new_state_dict

argparse = argparse.ArgumentParser()
argparse.add_argument('--input_path', type=str, help='Path to the adapter ckp', required=True)
argparse.add_argument('--output_path', type=str, help='Path to the output model', required=True)
argparse.add_argument('--base_model', type=str, help='Path to the base_model', default = "Qwen2.5-VL-7B-Instruct", required=False)
argparse.add_argument('--dino_path', type=str, help='Path to the dino ckp', default = "dino_vitl.pth", required=False)
args = argparse.parse_args()



print('Loading model...')
base_model_path = args.base_model
print(f"base model:{base_model_path}")
model = Qwen2_5_VL_Nautilus_ForConditionalGeneration.from_pretrained(
            base_model_path,
            cache_dir=None,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
# Dino
dino_path = args.dino_path
if "nautilus" in args.input_path.lower():
    print("Loading dino encoder")
    model.init_nautilus_model(dino_path, dino_only=True)

# load lora model and merge

init_kwargs = {
    "subfolder": None,
    "offload_folder": "offload",
    "cache_dir": None,
    "revision": "main",
    "token": None,
}

adapter = args.input_path

model: "LoraModel" = PeftModel.from_pretrained(model, adapter, **init_kwargs)

model = model.merge_and_unload()

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        cache_dir=None,
        model_max_length=8192,
        padding_side="right",
        use_fast=False,
    )

print('Saving model...')
model.save_pretrained(save_directory = args.output_path,
                    max_shard_size = f"5GB",
                    safe_serialization = True)
print('Saving tokenizer...')
tokenizer.save_pretrained(args.output_path)
print('Done!')
