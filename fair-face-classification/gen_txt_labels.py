"""Generate text embeddings and save to .pt file
"""
import json
import torch
from emb_generator import EmbGenerator

ROOT = "/home/lazye/Documents/ufrgs/mcs/clip/clip-bias-explore/\
fair-face-classification"
JSON_PATH = ROOT + "/data/labels"
EMBS_PATH = ROOT + "/data/embeddings"
embedder = EmbGenerator()
embedder.model_setup("ViT-B/16")

with open(JSON_PATH+"/caption_rad.json", encoding="utf-8") as f:
    data = json.load(f)

prompts = list(data.values())
labels = list(data.keys())
txt_fts = embedder.generate_text_embeddings(prompts)
torch.save(txt_fts, EMBS_PATH+"/age_race_gender.pt")
