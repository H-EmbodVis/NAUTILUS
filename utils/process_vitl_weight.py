import torch
import argparse

paser = argparse.ArgumentParser()
paser.add_argument('--dav2-vitl', type=str, default='Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth')
paser.add_argument('--dinov2-vitl', type=str, default='checkpoints/dino-vitl/dino_vitl.pth')
args = paser.parse_args()
  

dav2_weights = torch.load(args.dav2_vitl, map_location='cpu')
dino_vitl_weight = {}
for k,v in dav2_weights.items():
    # Exclude the depth_head weights and rename the remaining parameters accordingly
    if "pretrained" in k:
        dino_vitl_weight[k.replace("pretrained.","")] = v

torch.save(dino_vitl_weight, args.dinov2_vitl)