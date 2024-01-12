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

    def get_similarities(self, img_embs, txt_embs):
        """Grab similarity between text and image embeddings."""
        image_features = torch.from_numpy(img_embs).to(self.device)
        similarity = 100.0 * image_features @ txt_embs.T

        return similarity

    def get_synms_winner(self, sims):
        """transform similarity tensor into numpy array,
            then grab the index of the highest element"""
        np_sims = sims.cpu().numpy()
        np_loc = np.where(np_sims[0] == np_sims.max())
        return np_loc[0][0]

    def run_clip_classifier(self, img_emb, txt_embs):
        """Run txt_embs by CLIP to choose the closest one"""
        sims = self.get_similarities(img_emb, txt_embs)
        sims_max = sims.softmax(dim=-1)
        values, indices = sims_max[0].topk(len(sims_max[0]))
        scores = []
        for value, index in zip(values, indices):
            scores.append(
                (txt_embs[index], round(100 * value.item(), 2)))
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
    MODEL_NAME = "RN50"
    ROOT = "/home/lazye/Documents/ufrgs/mcs/clip/bias-explore/fair-face-red-circle"
    WOMAN_EMBS_PATH = ROOT+"/data/woman_embeddings.csv"
    MAN_EMBS_PATH = ROOT+"/data/man_embeddings.csv"
    FFACE_PATH = ROOT+"/data/fface_train.csv"
    GENDER_JSON = ROOT+"/data/gender-synms.json"

    sym_eval = SynmsEval(MODEL_NAME)

    woman_img_embs = pd.read_pickle(WOMAN_EMBS_PATH)
    man_img_embs = pd.read_pickle(MAN_EMBS_PATH)

    woman_txt_embs = torch.load(ROOT+'/data/woman-synms-embs.pt')
    man_txt_embs = torch.load(ROOT+'/data/man-synms-embs.pt')

    gender_txt_embs = torch.load(ROOT+'/data/gender-synms-embs.pt')

    with open(GENDER_JSON, encoding='utf-8') as json_data:
        gender_classes = json.load(json_data)['synms']

    fface_df = pd.read_csv(FFACE_PATH)

    woman_res = {}
    for _, emb in woman_img_embs.iterrows():
        name = emb['file']
        print(name)
        img_features = emb['embeddings']

        sims = sym_eval.get_similarities(img_features, gender_txt_embs)
        preds = gender_classes[sym_eval.get_synms_winner(sims)]
        print(preds)
        woman_res[name] = preds

    man_res = {}
    for _, emb in man_img_embs.iterrows():
        name = emb['file']
        print(name)
        img_features = emb['embeddings']

        sims = sym_eval.get_similarities(img_features, gender_txt_embs)
        preds = gender_classes[sym_eval.get_synms_winner(sims)]
        print(preds)
        man_res[name] = preds

    joint_res = {}
    joint_res.update(woman_res)
    joint_res.update(man_res)

    final_score_df = sym_eval.process_results(joint_res)
    final_df = fface_df.set_index('file').join(
        final_score_df.set_index('file'))

    final_df.rename(
        columns={'predictions': 'synms_gender_preds'}, inplace=True)
    final_df.to_csv(ROOT+'/data/full_synms_gender_fface.csv')
