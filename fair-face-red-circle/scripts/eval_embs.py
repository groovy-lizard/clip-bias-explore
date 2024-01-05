"""Evaluate embbeddings given target labels"""
import torch
import clip
import pandas as pd


# TODO: refactor so evaluator only handles embeddings and dataframe logics
class Evaluator:
    """
    Embeddings evaluator, the constructor receives the chosen CLIP model,
    a list of target class labels, and the paths to embeddings
    + core dataset .csv files
    """

    def __init__(self, model, classes, embs_path, face_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(model)
        self.classes = classes
        self.emb_df = pd.read_pickle(embs_path)
        self.fface_df = pd.read_csv(face_path)

    def load_model(self, model):
        """Initial loading of CLIP model.
        list of available models: 'RN50', 'RN101', 'RN50x4', 'RN50x16'"""
        available_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16']

        if model in available_models:
            print(f'Loading model: {model}')
            clip_model = model
        else:
            print(f'{model} unavailable! Falling back to default model: RN50')
            clip_model = available_models[0]

        model, _ = clip.load(
            clip_model, device=self.device, jit=False)

        print(f'Done! Model loaded to {self.device} device')
        return model

    def eval_embds(self):
        """Softmax of pairs of the embeddings and each target class label"""
        results = {}

        text_inputs = torch.cat(
            [clip.tokenize(f"a photo of a {c}") for c in self.classes]
        ).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)

        text_features /= text_features.norm(dim=-1, keepdim=True)

        for _, emb in self.emb_df.iterrows():
            name = emb['file']
            image_features = emb['embeddings']
            image_features = torch.from_numpy(image_features).to(self.device)
            similarity = (100 * image_features @
                          text_features.T).softmax(dim=-1)

            values, indices = similarity[0].topk(len(similarity[0]))
            scores = []
            for value, index in zip(values, indices):
                scores.append(
                    (self.classes[index], round(100 * value.item(), 2)))
            results[name] = scores

        return results

    def update_embs(self, embs_path):
        """Update loaded embeddings given a new .csv path"""
        self.emb_df = pd.read_pickle(embs_path)
        return 0

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

    def generate_final_df(self, score_df):
        """Join the winning class df with the original df"""
        new_df = self.fface_df.set_index(
            'file').join(score_df.set_index('file'))
        return new_df


if __name__ == "__main__":
    races = ['White', 'Latino_Hispanic', 'Indian', 'East Asian',
             'Black', 'Southeast Asian', 'Middle Eastern']
    MAN_EMBS_PATH = './man_embeddings.csv'
    WOMAN_EMBS_PATH = './woman_embeddings.csv'
    FFACE_PATH = './fface-train.csv'

    # class should'nt have to handle paths and whatnot
    evaluator = Evaluator("RN50", races, MAN_EMBS_PATH, FFACE_PATH)

    man_embs_results = evaluator.eval_embds()
    evaluator.update_embs(WOMAN_EMBS_PATH)  # !refactor this

    woman_embs_results = evaluator.eval_embds()
    joint_embs_results = {}
    joint_embs_results.update(man_embs_results)
    joint_embs_results.update(woman_embs_results)

    final_score_df = evaluator.process_results(joint_embs_results)
    final_race_fface_df = evaluator.generate_final_df(final_score_df)
    final_race_fface_df.rename(
        columns={'predictions': 'race_preds'}, inplace=True)
    final_race_fface_df.to_csv('new_fface.csv')
