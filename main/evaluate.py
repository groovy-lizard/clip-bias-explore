"""Evaluate synonyms text embeddings over FairFace validation images"""
import json
import torch
import clip
import numpy as np
import pandas as pd


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


def get_similarities(img, txts):
    """Grab similarity between text and image embeddings."""
    image_features = torch.from_numpy(img).to('cuda')
    similarity = 100.0 * image_features @ txts.T

    return similarity


def get_sims_dict(im_embs, txt_prompts, t_embs):
    """Generate dictionary with filename and similarities
    scores between text prompts"""
    final_dict = {}
    for _, emb in im_embs.iterrows():
        name = emb['file']
        img_features = emb['embeddings']
        img_sims = get_similarities(img_features, t_embs)
        s_dict = {}
        for label, score in zip(txt_prompts, img_sims[0]):
            s_dict[label] = score.cpu().numpy().item()
        final_dict[name] = s_dict
    # TODO: fix naming convention
    with open('similarities.json', 'w', encoding='utf-8') as ff:
        json.dump(obj=final_dict, fp=ff, indent=4)
    return final_dict


def get_top_synm(final_dict):
    """Grab most similar synonym"""
    files = final_dict.keys()
    wins = []
    for val in final_dict.values():
        scores_list = list(val.values())
        label_list = list(val.keys())
        np_scores = np.asarray(scores_list)
        windex = np.where(np_scores == np_scores.max())[0][0]
        wins.append(label_list[windex])

    top_synm_dict = {'file': files, 'gender_preds': wins}
    return pd.DataFrame(data=top_synm_dict)


def get_sum_synms(final_dict, man_prompts):
    """Ensemble over avg sum of similarities
    between male and female synms"""
    files = final_dict.keys()
    preds = []

    for _, val in final_dict.items():
        man_score = 0
        woman_score = 0
        for k, v in val.items():
            if k in man_prompts:
                man_score += v
            else:
                woman_score += v
        preds.append('Male' if man_score > woman_score else 'Female')

    sum_dict = {'file': files, 'gender_preds': preds}
    return pd.DataFrame(data=sum_dict)


def synm_to_gender(synm, man_prompts):
    """Mapper function to eval between Male and Female
    synonyms"""
    if synm in man_prompts:
        return 'Male'
    else:
        return 'Female'


def generate_final_df(f_df, score_df):
    """Join the winning class df with the original df"""
    new_df = f_df.set_index(
        'file').join(score_df.set_index('file'))
    new_df.drop(columns=['service_test'], inplace=True)
    return new_df


def save_df(df, out):
    """Save df to csv"""
    print(f"saving df to {out}")
    df.to_csv(out)
    print("Done!")
    return 0


def map_synm_to_gender(df, man_prompts):
    """Use sub-set of man synms to evaluate and map
    synms to Male or Female"""
    new_df = df.copy()
    new_df['synm'] = new_df['gender_preds']
    new_df['gender_preds'] = df['gender_preds'].map(
        lambda x: synm_to_gender(x, man_prompts))
    return new_df


if __name__ == "__main__":
    ROOT = "/home/lazye/Documents/ufrgs/mcs/clip/clip-bias-explore/\
fair-face-classification"
    LABELS_PATH = ROOT + "/data/labels"
    EMBS_PATH = ROOT + "/data/embeddings"
    RESULTS_PATH = ROOT + "/data/results"

    vit_model = model_setup('ViT-B/16')
    img_embs = pd.read_pickle(EMBS_PATH+"/fface_val_img_embs.pkl")
    txt_embs = torch.load(EMBS_PATH+"/age_race_gender.pt")
    fface_df = pd.read_csv(ROOT+"/data/fface_val.csv")

    with open(LABELS_PATH+"/caption_rad.json", encoding='utf-8') as f:
        data = json.load(f)

    prompt_list = list(data.values())
    man_p = prompt_list[:32]

    sims_dict = get_sims_dict(img_embs, prompt_list, txt_embs)
    # sum_df = get_sum_synms(sims_dict, man_p)
    # final_sum_df = generate_final_df(fface_df, sum_df)
    # save_df(final_sum_df, RESULTS_PATH+"/arg_sum_synms.csv")

    # top_df = get_top_synm(final_dict=sims_dict)
    # bin_top_df = map_synm_to_gender(top_df, man_p)
    # final_top_df = generate_final_df(fface_df, bin_top_df)
    # save_df(final_top_df, RESULTS_PATH+"/arg_top_synms.csv")
