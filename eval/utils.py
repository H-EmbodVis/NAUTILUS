import re
import torch
import json
from PIL import Image

image_token_id = 151655
task_tags = {"Image_caption":"0","Grounding":"1","Region_caption":"2","VQA":"3","Fishnet_Classification":"4","Detection":"5","Counting":"6","Region_Classification":"7"}
tag2task = {v : k for k,v in task_tags.items()}
def sortbyid(conversations: list):
    '''
    conversations : item in conversations must have attribute "id"
    '''
    sorted_dict = {t : []for t in task_tags}
    for convs in conversations:
        sorted_dict[tag2task[convs["id"][1]]].append(convs)

    # Remove empty items
    sorted_dict = {k: v for k, v in sorted_dict.items() if len(v) > 0}
    return sorted_dict

def scale_bboxes_in_text(text: str, w_scale: float, h_scale: float) -> str:

    def scale_bbox(match):
        x1 = int(match.group(1))
        y1 = int(match.group(2))
        x2 = int(match.group(3))
        y2 = int(match.group(4))
        new_x1 = round(x1 * w_scale)
        new_y1 = round(y1 * h_scale)
        new_x2 = round(x2 * w_scale)
        new_y2 = round(y2 * h_scale)
        return f"[{new_x1}, {new_y1}, {new_x2}, {new_y2}]"

    # match format like [x1, y1, x2, y2]
    pattern = r"\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]"
    scaled_text = re.sub(pattern, scale_bbox, text)
    return scaled_text

def get_grid_thw(processor, image_file):
    image = Image.open(image_file).convert("RGB")
    width, height = image.size
    visual_processed = processor.preprocess(image, return_tensors="pt")
    grid_thw = visual_processed["image_grid_thw"][0]
    return grid_thw, width, height

def double_image_tokens(inputs: dict, image_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Input:
        input_ids: torch.Tensor, shape (1, N), input token ids.
        image_token_id: int, image token id.
    
    Return:
        torch.Tensor, shape (1, N'), Doubled image token.
    """

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    input_ids = input_ids.squeeze(0) 
    attention_mask = attention_mask.squeeze(0)
    new_ids, new_mask = [], []

    for token,mask in zip(input_ids, attention_mask):
        new_ids.append(token.item())
        new_mask.append(mask.item())
        if token.item() == image_token_id:
            new_ids.append(token.item())
            new_mask.append(mask.item())

    return torch.tensor(new_ids, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0), torch.tensor(new_mask, dtype=attention_mask.dtype, device=attention_mask.device).unsqueeze(0)