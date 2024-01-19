"""Embeddings generator using CLIP image or text feature extractor"""
import clip
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm


def model_setup(model):
    """Initial loading of CLIP model."""

    available_models = clip.available_models()

    if model in available_models:
        print(f'Loading model: {model}')
        chosen_model = model
    else:
        print(f'{model} unavailable! Using default model: ViT-L/14@336px')
        chosen_model = available_models[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, pps = clip.load(chosen_model, device=device, jit=False)

    print(f'Done! Model loaded to {device} device')
    return model, pps


def generate_img_ebs_df(df, model, preprocess, device, path):
    """Generate image embeddings using CLIP model"""
    files = []
    embs = []

    for file in tqdm(df):
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


def generate_txt_embs(txts, model, device):
    """Generate text embeddings using CLIP model"""
    text_inputs = torch.cat(
        [clip.tokenize(c) for c in txts]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features


if __name__ == "__main__":
    fface_df = pd.read_csv("../data/fface_val.csv")
    clip_model, preprocessor = model_setup('ViT-B/16')
    VAL_PATH = '/home/lazye/Documents/ufrgs/mcs/datasets/FairFace/'
    # f = open('../data/labels.json', encoding='utf-8')
    # labels = json.load(f)
    # f.close()
    # labels = list(labels.values())
    # txt_embs = generate_txt_embs(labels, clip_model, 'cuda')
    # torch.save(txt_embs, '../data/original-gender-race-labels.pt')
    img_embs_df = generate_img_ebs_df(
        fface_df['file'], clip_model, preprocessor, 'cuda', VAL_PATH)
    img_embs_df.to_pickle('../data/fface_val_img_embs.pkl')