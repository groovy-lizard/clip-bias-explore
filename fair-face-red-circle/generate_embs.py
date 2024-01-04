import clip
import torch
import pandas as pd
import numpy as np
from PIL import Image


def model_setup(model):
    """Initial loading of CLIP model.
    list of available models: 'RN50', 'RN101', 'RN50x4', 'RN50x16'"""
    available_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16']

    if model in available_models:
        print(f'Loading model: {model}')
        clip_model = model
    else:
        print(f'{model} unavailable! Falling back to default model: RN50')
        clip_model = available_models[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, preprocess = clip.load(clip_model, device=device, jit=False)

    print(f'Done! Model loaded to {device} device')
    return model, preprocess


def generate_embeddings_dataframe(df, model, preprocess, device, path):
    """Generate image embeddings using CLIP model"""
    files = []
    embs = []

    for file in df:
        img_path = path + file
        img = Image.open(img_path)
        img_input = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(img_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        files.append(file)
        embs.append(image_features.cpu().numpy())

    d = {'file': files, 'embeddings': embs}

    df_out = pd.DataFrame(data=d)
    return df_out
