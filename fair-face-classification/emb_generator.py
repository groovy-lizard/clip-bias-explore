"""
CLIP embeddings generator
Tools include the generation of image and text embeddings,
"""

import sys
import torch
import clip
import pandas as pd
from tqdm import tqdm
from PIL import Image


class EmbGenerator:
    """Embeddings generator main class. Instanced without parameters,
        but should call model_setup before use"""

    def __init__(self):
        self.model = None
        self.pps = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def model_setup(self, model_name):
        """Initial loading of CLIP model.

        Keyword arguments:
        model_name (str) -- model name chosen from clip.available_models()"""

        available_models = clip.available_models()

        if model_name in available_models:
            print(f'Loading model: {model_name}')
            chosen_model = model_name
        else:
            print(
                f'{model_name} unavailable!')
            sys.exit(-1)

        self.model, self.pps = clip.load(
            chosen_model, device=self.device, jit=False)

        print(f'Done! Model loaded to {self.device} device')

    def generate_image_embeddings(self, img_list):
        """Generate image embeddings and returns a pandas dataframe
        with { name of file: img_emb }

        Keyword arguments:
        img_list (list[str]) -- a list of image paths
        """

        files = []
        embs = []

        for file_name in tqdm(img_list):
            img = Image.open(file_name)
            img_input = self.pps(img).unsqueeze(0).to(self.device)

            with torch.nograd():
                image_features = self.model.encode_image(img_input)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            files.append(file_name)
            embs.append(image_features.cpu().numpy())

        d = {'file': files, 'embeddings': embs}

        df_out = pd.DataFrame(data=d)
        return df_out

    def generate_text_embeddings(self, txt_list):
        """Generate text embeddings and returns a tokenized torch tensor

        Keyword arguments:
        txt_list (list[str]) -- a list of text prompts"""

        text_inputs = torch.cat(
            [clip.tokenize(c) for c in txt_list]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features
