"""This module aims to evaluate different label synonyms
and choose the best one, in order to compete agains another target label"""
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load(model, device=self.device, jit=False)

    def get_similarities(self, img_embs, classes):
        """Grab similarity between classes and image embeddings."""
        image_features = torch.from_numpy(img_embs).to(self.device)

        text_inputs = torch.cat(
            [clip.tokenize(f"a photo of a {c}") for c in classes]
        ).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)

        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = 100.0 * image_features @ text_features.T
        return similarity

    def get_synms_winner(self, sims):
        """transform similarity tensor into numpy array,
            then grab the index of the highest element"""
        np_sims = sims.cpu().numpy()
        np_loc = np.where(np_sims[0] == np_sims.max())
        return np_loc[0][0]

    def run_clip_classifier(self, img_emb, classes):
        """Run classes by CLIP to choose the closest one"""
        sims = self.get_similarities(img_emb, classes)
        sims_max = sims.softmax(dim=-1)
        values, indices = sims_max[0].topk(len(sims_max[0]))
        scores = []
        for value, index in zip(values, indices):
            scores.append(
                (classes[index], round(100 * value.item(), 2)))
        return scores

    def process_results(self, results):
        """Grab the file and the winning class from results
          and returns a dataframe"""
        score_dict = {}
        files = []
        predicts = []

        for key, value in results.items():
            files.append(key)
            predicts.append(value[0][0])

        score_dict['file'] = files
        score_dict['predictions'] = predicts

        scores_df = pd.DataFrame(data=score_dict)
        return scores_df


if __name__ == "__main__":
    MODEL_NAME = "RN50"
    WOMAN_EMBS_PATH = "../data/woman_embeddings.csv"
    MAN_EMBS_PATH = "../data/man_embeddings.csv"
    woman_labels = ['girl', 'lady', 'woman', 'young_woman', 'matriarch',
                    'female_person', 'female', 'girlfriend', 'wife',
                    'adult_female', 'young_lady', 'broad', 'madam',
                    'lady_friend']

    man_labels = ['boy', 'gentleman', 'man', 'dude', 'patriarch', 'husband',
                  'sir', 'guy', 'male_person', 'adult_male', 'male',
                  'young_man', 'boyfriend']

    sym_eval = SynmsEval(MODEL_NAME)
    woman_embs_df = pd.read_pickle(WOMAN_EMBS_PATH)
    man_embs_df = pd.read_pickle(MAN_EMBS_PATH)
    res = {}

    for _, emb in woman_embs_df.iterrows():
        name = emb['file']
        print(name)
        image_features = emb['embeddings']

        woman_sims = sym_eval.get_similarities(image_features, woman_labels)
        woman_win = sym_eval.get_synms_winner(woman_sims)

        man_sims = sym_eval.get_similarities(image_features, man_labels)
        man_win = sym_eval.get_synms_winner(man_sims)

        preds = sym_eval.run_clip_classifier(
            image_features, [woman_win, man_win])

        res[name] = preds

    res_df = sym_eval.process_results(res)
    print(res_df.head())
