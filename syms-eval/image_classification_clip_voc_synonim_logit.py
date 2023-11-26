# -*- coding: utf-8 -*-
"""
Compares mAP of vanilla CLIP with a synonym-based strategy

"""

import os
import clip
import torch
from glob import glob
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import pickle
# from torchvision.datasets import CIFAR100

#
#  mAP files from DASSL
#


def evaluator(gt, det, class_names, verbose=True):

    map, ap = mAP(np.array(gt), np.array(det))
    print('\nmAP: %1.2f' % map)
    print('\nPer class AP values')
    for i in range(20):
        print(class_names[i] + ': %.2f' % ap[i])


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    # print('shape', targs.shape)
    # print(preds.shape)
    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    map = 100 * ap.mean()
    ap = 100 * ap
    return map, ap


def read_synonyms(file_name):
    #
    # Reads list of synonims.
    # Produces a list of synonims for the 20 VOC classes,
    # plus background (maybe)
    #
    file = open(file_name, 'r')
    data = file.readlines()
    file.close()

    #
    #  Processes txt file
    #
    names = []
    for entry in data:
        #
        #  removes {  } and \n
        #
        entry = entry[1:-2].replace("'", "").split(',')
        names.append(entry)

    return names


def generate_query(class_names):
    #
    #  Generates text queries (canonical class names)
    #
    text_inputs = torch.cat(
        [clip.tokenize(f"A photo of a {c}") for c in class_names]).to(device)
    return text_inputs


def generate_query_synonym(query_names, addplurals=False):
    #
    #  Generates text queries
    #

    prompts = []
    labels = []
    #
    #  Scans all categories
    #
    for i, query in enumerate(query_names):
        #
        #  Scans synonyms inj each category
        #
        for name in query:
            labels.append(i)
            prompts.append(clip.tokenize("A photo of a %s" % name))
            if addplurals:
                labels.append(i)
                prompts.append(clip.tokenize("A photo of a %ss" % name))

    text_inputs = torch.cat(prompts)
    return text_inputs, labels


def scan_dataset(filelist):
    labels = []
    for file in tqdm(filelist):
        xmlfile = ann_dir + file.split('\\')[-1].replace('.jpg', '.xml')
        all_labels = pasrse_voc(xmlfile)
        labels.append(all_labels)
    return labels


def pasrse_voc(in_file):
    all_labels = []
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if not (cls in all_labels):
            all_labels.append(cls)

    return all_labels


def multi_label_synonym(image_features, text_features,
                        class_names, labels, tau=100.0):
    #
    #  Gets logits using the best synonim for each category
    #

    #
    #  query_names contain the prompt embeddings,
    # and the corresponding class labels are in varaible labels
    #

    #
    #  Computes cosine similarity with temperature tau
    #
    scores = (tau*image_features @ text_features.T)

    #
    # Performs max-pooling within each category
    #

    nclasses = len(class_names)

    scores_class = np.zeros(nclasses)
    for i in range(nclasses):
        scores_class[i] = scores[0, np.array(labels) == i].max()

    return scores_class


if __name__ == '__main__':

    LoadFile = True
    SaveFile = False
    ShowResults = False

    #
    #  Dataset and available CLIP models
    #
    available_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16',
                        'RN50x64', 'ViT-B/32', 'ViT-B/16',
                        'ViT-L/14', 'ViT-L/14@336px']
    available_datasets = ['VOC2007', 'VOC2007_test', 'VOC2012']
    dataset = available_datasets[1]

    #
    #  Directories and files
    #
    image_dir = '/home/lazye/Documents/ufrgs/mcs/pascalVOC/2007/JPEGImages/'
    class_txt = 'voc_class_names.txt'
    ann_dir = '/home/lazye/Documents/ufrgs/mcs/pascalVOC/2007/Annotations/'
    # txt files with the list of synonyms
    synonym_file = 'synonyms_VOC_v1background.txt'

    #
    # Folder for saving results
    #
    version = synonym_file.split('_')[-1].split('.')[0]
    savedir = 'synonym/' + version
    if not (os.path.isdir(savedir)):
        os.mkdir(savedir)

    #
    #  Loads GT annotations (from TAI-DPT)
    #
    all_gt = np.load(dataset + '_gt.npy')

    # batch processing of different models
    for modelname in [available_models[6]]:
        modelname_pruned = modelname.replace('/', '')

        #
        #  Reads class names for dataset
        #
        with open(class_txt) as file_class:
            class_names = file_class.read().splitlines()
        #
        #  removes backgound class in the beggining and adds at the end
        #
        class_names = class_names[1:]
        class_names.append('background')

        #
        #  Queries for classes and synonyms
        #
        query_names = read_synonyms(synonym_file)
        num_classes = len(query_names)
        filelist = glob(image_dir + '*.jpg')
        num_files = len(filelist)

        all_det_labels_canonical = np.zeros((num_files, num_classes))
        all_det_labels_synonym = np.zeros((num_files, num_classes))

        if LoadFile:
            #
            #  Loads pre-computed logits
            #
            file = open(savedir + '/' + modelname_pruned +
                        dataset + '_logits_synonym.dat', 'rb')
            all_det_labels_synonym = pickle.load(file)
            file.close()

            file = open(savedir + '/' + modelname_pruned +
                        dataset + '_logits_canonical.dat', 'rb')
            all_det_labels_canonical = pickle.load(file)
            file.close()

        else:
            #
            #  Computes logits from text and image embeddings
            #

            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load(modelname, device)

            #
            #  Gets temperature trained with model
            # (see https://github.com/openai/CLIP/blob/main/clip/model.py)
            #
            tau = float(model.logit_scale.exp().detach())

            #
            #  VOC files (images)
            #
            filelist = glob(image_dir + '*.jpg')
            num_files = len(filelist)

            #
            #  Arrays for storing logits with canonical names and synonyms
            #
            all_det_labels_canonical = np.zeros((num_files, num_classes))
            all_det_labels_synonym = np.zeros((num_files, num_classes))

            #
            #  Generates initial text prompts
            #

            # Queries with synonyms and canonical classes
            text_inputs_synonym, labels = generate_query_synonym(
                query_names, addplurals=False)
            text_inputs_canonical = generate_query(class_names)

            # Calculates and normalizes features
            with torch.no_grad():
                text_features_canonical = model.encode_text(
                    text_inputs_canonical)
                text_features_synonym = model.encode_text(text_inputs_synonym)

            text_features_canonical /= text_features_canonical.norm(
                dim=-1, keepdim=True)
            text_features_synonym /= text_features_synonym.norm(
                dim=-1, keepdim=True)

            #
            #  Loads all images
            #
            for ind, file in enumerate(tqdm(filelist)):
                image = Image.open(file)
                xmlfile = ann_dir + \
                    file.split('/')[-1].replace('.jpg', '.xml')
                img = image

                #
                #  Gets embeddings
                #
                image_input = preprocess(img).unsqueeze(0).to(device)

                # Calculates and normalizes image embeddings
                with torch.no_grad():
                    image_features = model.encode_image(image_input)

                image_features /= image_features.norm(dim=-1, keepdim=True)

                #
                #  Gets logits (inner product between image and text
                # embeddings scaled by temperature)
                #
                syn_logits = multi_label_synonym(
                    image_features, text_features_synonym,
                    class_names, labels, tau)
                can_logits = (tau * image_features @
                              text_features_canonical.T)[0].numpy()


# image_id = int(file.split('\\')[-1].split('.')[0])
# image_labels = gts = np.load(dai_dir + 'label_%06d.npy'%image_id)

                #
                #  Stores logits for all images
                #
                all_det_labels_synonym[ind] = syn_logits
                all_det_labels_canonical[ind] = can_logits

            if SaveFile:
                #
                #  Saves logits
                #
                file = open(savedir + '/' + modelname_pruned +
                            dataset + '_logits_synonym.dat', 'wb')
                pickle.dump(all_det_labels_synonym, file)
                file.close()

                file = open(savedir + '/' + modelname_pruned +
                            dataset + '_logits_canonical.dat', 'wb')
                pickle.dump(all_det_labels_canonical, file)
                file.close()

        #
        # Evaluates results - mAP
        #

        print('Baseline: %s\nDataset:%s\n' % (modelname, dataset))
        print('Results with original CLIP')
        evaluator(all_gt,
                  all_det_labels_canonical[:, :20], class_names, verbose=True)
        print('\nResults with synonym-CLIP')
        evaluator(all_gt,
                  all_det_labels_synonym[:, :20], class_names, verbose=True)
