import sys

from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from model import BERT
from data import MyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_embeddings(texts, embeddings, plot_name):
    """
    Uses MDS to plot embeddings (and its respective sentence) in 2D space.

    Inputs:
    - texts: A list of strings, representing the words
    - embeddings: A 2D numpy array, [num_sentences x embedding_size],
        representing the relevant word's embedding for each sentence
    """
    embeddings = embeddings.astype(np.float64)
    mds = MDS(n_components=2)
    embeddings = mds.fit_transform(embeddings)

    plt.figure(1)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], color='navy')
    for i, text in enumerate(texts):
        plt.annotate(text, (embeddings[i, 0], embeddings[i, 1]))
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    plt.savefig(plot_name, dpi=100)


def embedding_analysis(model, experiment, train_set, test_set, epoch=None):
    """
    Create embedding analysis image for each list of polysemous words and
    upload them to comet.ml.

    Inputs:
    - model: Trained BERT model
    - experiment: comet.ml experiment object
    - train_set: train dataset
    - test_set: test dataset
    """
    polysemous_words = {
        "figure": ["figure", "figured", "figures"],
        "state": ["state", "states", "stated"],
        "bank": ["bank", "banks", "banked"]
    }

    word2id = train_set.word2id
    with torch.no_grad():
        for key in polysemous_words:
            # TODO: Find all instances of sentences that have polysemous words.
            relevent_seqs = list()
            texts = list()
            for seq in train_set.all_data:
                for word in polysemous_words[key]:
                    try:
                        wid = word2id[word]
                        index = seq.index(wid)
                        relevent_seqs.append((seq, index))
                        texts.append(word)
                    except (KeyError, ValueError):
                        continue
            for seq in test_set.all_data:
                for word in polysemous_words[key]:
                    try:
                        wid = word2id[word]
                        index = seq.index(wid)
                        relevent_seqs.append((seq, index))
                        texts.append(word)
                    except (KeyError, ValueError):
                        continue

            # TODO: Give these sentences as input, and obtain the specific word
            #       embedding as output.
            embedding = list()
            for seq, index in relevent_seqs:
                embed = model.get_embeddings( torch.LongTensor(seq).unsqueeze(0).to(device) )[0][index].cpu().numpy()
                embedding.append(embed)
            embedding = np.array(embedding)

            # TODO: Use the plot_embeddings function above to plot the sentence
            #       and embeddings in two-dimensional space
            name = key
            if epoch is not None:
                name = key + "-" + str(epoch)
            plot_embeddings(texts, embedding, "%s.png"%name)

            # TODO: Save the plot as "{word}.png"
            experiment.log_image("%s.png"%name)
            plt.close()
