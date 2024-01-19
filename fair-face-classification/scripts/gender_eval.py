"""This module aims to evaluate different label synonyms
and choose the best one, in order to compete agains another target label"""
import json
import torch
import clip
import numpy as np
import pandas as pd


class SynmsEval:
    """Synonym evaluator
        Helper functions for evaluating best synonyms,
        i.e. the one closest with the target text label embedding.
    """

    def __init__(self, model):
        print(f'Loading model {model}...')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load(model, device=self.device, jit=False)
        print(f'Done! Model loaded into {self.device}')

    def get_similarities(self, img, txts):
        """Grab similarity between text and image embeddings."""
        image_features = torch.from_numpy(img).to(self.device)
        similarity = 100.0 * image_features @ txts.T

        return similarity

    def get_synms_winner(self, sims):
        """transform similarity tensor into numpy array,
            then grab the index of the highest element"""
        np_sims = sims.cpu().numpy()
        np_loc = np.where(np_sims[0] == np_sims.max())
        return np_loc[0][0]

    def run_clip_classifier(self, img_emb, txts):
        """Run txts by CLIP to choose the closest one"""
        sms = self.get_similarities(img_emb, txts)
        sms_max = sms.softmax(dim=-1)
        values, indices = sms_max[0].topk(len(sms_max[0]))
        scores = []
        for value, index in zip(values, indices):
            scores.append(
                (txts[index], round(100 * value.item(), 2)))
        return scores

    def process_results(self, results):
        """Grab the file and the winning class from results
          and returns a dataframe"""
        score_dict = {}
        files = []
        predicts = []

        for key, value in results.items():
            files.append(key)
            predicts.append(value)

        score_dict['file'] = files
        score_dict['predictions'] = predicts

        scores_df = pd.DataFrame(data=score_dict)
        return scores_df


if __name__ == "__main__":
    MODEL_NAME = "ViT-B/16"
    ROOT = "/home/lazye/Documents/ufrgs/mcs/" + \
        "clip/bias-explore/fair-face-classification"
    IMG_EMBS_PATH = ROOT+"/data/fface_val_img_embs.pkl"
    FFACE_PATH = ROOT+"/data/fface_val.csv"
    LABELS_JSON = ROOT+"/data/raw_gender_labels.json"

    sym_eval = SynmsEval(MODEL_NAME)

    img_embs = pd.read_pickle(IMG_EMBS_PATH)

    txt_embs = torch.load(ROOT+'/data/raw-gender-labels.pt')

    with open(LABELS_JSON, encoding='utf-8') as json_data:
        data = json.load(json_data)
        fface_classes = list(data.keys())
        fface_prompts = list(data.values())

    fface_df = pd.read_csv(FFACE_PATH)

    res = {}
    for _, emb in img_embs.iterrows():
        name = emb['file']
        print(name)
        img_features = emb['embeddings']

        sims = sym_eval.get_similarities(img_features, txt_embs)
        preds = fface_classes[sym_eval.get_synms_winner(sims)]
        res[name] = preds

    final_score_df = sym_eval.process_results(res)
    final_df = fface_df.set_index('file').join(
        final_score_df.set_index('file'))

    final_df.rename(
        columns={'predictions': 'gender_preds'}, inplace=True)
    final_df.to_csv(ROOT+'/data/raw_gender_preds_fface.csv')
