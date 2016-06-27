import numpy as np
import pandas as pd
from collections import defaultdict

def create_unk_labeled_instances_row(sentence, q1, q2, q3, q4, img_file):
    if q2 == 'man-made':
        new_q3 = [l for l in q3map.values()
                  if l not in ['NA', 'other_unclear']]
        new_sentence = [sentence for _ in range(len(new_q3))]
        new_q1 = [q1 for _ in range(len(new_q3))]
        new_q2 = [q2 for _ in range(len(new_q3))]
        new_q4 = ['NA' for _ in range(len(new_q3))]
        new_img_file = [img_file for _ in range(len(new_q3))]

    elif q2 == 'natural':
        new_q4 = [l for l in q4map.values()
                  if l not in ['NA', 'other_unclear']]
        new_sentence = [sentence for _ in range(len(new_q4))]
        new_q1 = [q1 for _ in range(len(new_q4))]
        new_q2 = [q2 for _ in range(len(new_q4))]
        new_q3 = ['NA' for _ in range(len(new_q4))]
        new_img_file = [img_file for _ in range(len(new_q4))]

    return pd.DataFrame({'sentence': new_sentence, 'q1': new_q1,
                        'q2': new_q2, 'q3': new_q3, 'q4': new_q4,
                        'img_file': new_img_file})

def create_unk_labeled_instances(df):
    """Create a df by mapping any other/unclear image to every possible
    leaf-level label."""
    unk_df = df[(df.q3 == 'other_unclear') | (df.q4 == 'other_unclear')]
    new_dfs = []
    for _, row in unk_df.iterrows():
        new_dfs.append(create_unk_labeled_instances_row(*row))
    unk_labeled_df = pd.concat(new_dfs)
    return unk_labeled_df

def find_unlabeled_sents(captions_file, df):
    """
    Add sentences that match keywords for each label to df
    """
    #TODO incomplete
    body_of_water_words = ["beach", "lake", "ocean", "sea"]
    in_work_ed_words = ["office", "classroom", "in a shop", "in a store"]
    keywords = {'outdoors/man-made/body_of_water': body_of_water_words,
                'indoors/man-made/work_education': in_work_ed_words}

    captions = defaultdict(list)
    with open(captions_file, "r") as cf:
        for line in cf:
            img, sentence = line.strip().split('\t')
            img = img.split('#')[0]
            captions[img].append(sentence)

    match_sentences = []
    annotated, not_annotated = get_annotated_imgs()
    for img in not_annotated:
        match = False
        sentences = [s.strip().split() for s in captions[img]]
        for sentence in sentences:
            for (label, words) in keywords.items():
                if any(word in sentence for word in words) and not match:
                    match_sentences.append((label, sentences, img))
                    match = True

    return match_sentences

