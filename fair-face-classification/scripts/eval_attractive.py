"""Evaluate CLIP classification of attractiveness in human faces"""
from eval_embs import Evaluator

if __name__ == "__main__":
    man_labels = ['beautiful_man', 'man']
    woman_labels = ['beautiful_woman', 'woman']
    MAN_EMBS_PATH = '../data/man_embeddings.csv'
    WOMAN_EMBS_PATH = '../data/woman_embeddings.csv'
    FFACE_PATH = '../data/fface_train.csv'

    # class should'nt have to handle paths and whatnot
    man_evaluator = Evaluator("RN50", man_labels, MAN_EMBS_PATH, FFACE_PATH)
    woman_evaluator = Evaluator(
        "RN50", woman_labels, WOMAN_EMBS_PATH, FFACE_PATH)

    man_embs_results = man_evaluator.eval_embds()
    woman_embs_results = woman_evaluator.eval_embds()
    joint_embs_results = {}
    joint_embs_results.update(man_embs_results)
    joint_embs_results.update(woman_embs_results)

    final_score_df = woman_evaluator.process_results(joint_embs_results)
    final_fface_df = woman_evaluator.generate_final_df(final_score_df)

    final_fface_df.rename(
        columns={'predictions': 'beautiful_preds'}, inplace=True)
    final_fface_df.to_csv('../data/fface_beautiful_preds.csv')
