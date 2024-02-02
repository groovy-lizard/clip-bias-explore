"""Generate text embeddings and save to .pt file
change JSON_PATH for input, and EMBS_PATH for output"""
import json
import torch
from emb_generator import EmbGenerator

ROOT = "/home/lazye/Documents/ufrgs/mcs/clip/clip-bias-explore/\
fair-face-classification"
JSON_PATH = ROOT + "/data/labels"
EMBS_PATH = ROOT + "/data/embeddings"
clip_classifier = EmbGenerator()
clip_classifier.model_setup("ViT-B/16")

with open(JSON_PATH+"/bots_synms_prompts.json", encoding="utf-8") as f:
    data = json.load(f)

prompts = list(data.values())
labels = list(data.keys())
txt_fts = clip_classifier.generate_text_embeddings(prompts)
torch.save(txt_fts, EMBS_PATH+"/synms_labels.pt")
