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


if __name__ == "__main__":
    model_name = "RN50"
    img_embs_path = "../data/woman_embeddings.csv"
    label_group_1 = ['girl', 'lady', 'woman']
    label_group_2 = ['boy', 'gentleman', 'man']

    evaluator = SynmsEval(model_name)
    img_embs_df = pd.read_pickle(img_embs_path)
    chosen_img_emb = img_embs_df.iloc[0]['embeddings']

    g1_sims = evaluator.get_similarities(chosen_img_emb, label_group_1)
    g2_sims = evaluator.get_similarities(chosen_img_emb, label_group_2)
    g1_win = label_group_1[evaluator.get_synms_winner(g1_sims)]
    g2_win = label_group_2[evaluator.get_synms_winner(g2_sims)]

    final_classes = [g1_win, g2_win]
    print(evaluator.run_clip_classifier(chosen_img_emb, final_classes))
