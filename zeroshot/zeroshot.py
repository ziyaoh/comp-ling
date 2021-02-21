from comet_ml import Experiment
from preprocess import TranslationDataset, read_from_corpus
from model import Seq2Seq
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar

# TODO: Set hyperparameters
hyperparams = {
    "rnn_size": 128,  # assuming encoder and decoder use the same rnn_size
    "embedding_size": 128,
    "num_epochs": 2,
    "batch_size": 64,
    "learning_rate": 0.001, 
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, experiment, hyperparams, bpe):
    """
    Trains the model.

    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    :param bpe: is bpe dataset or not
    """
    # TODO: Define loss function and optimizer

    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    model = model.train()
    with experiment.train():
        # TODO: Write training loop
        for epoch in range(hyperparams["num_epochs"]):
            print("training for epoch {} / {}".format(epoch+1, hyperparams["num_epochs"]))
            for batch in tqdm(train_loader):
                enc_data = batch["enc_data"].to(device)
                dec_data = batch["dec_data"].to(device)
                label = batch["label"].to(device)
                enc_length = batch["enc_length"].to(device)
                dec_length = batch["dec_length"].to(device)

                logits = model(enc_data, dec_data, enc_length, dec_length)
                pred = torch.flatten(logits, start_dim=0, end_dim=1)
                label = torch.flatten(label)

                optimizer.zero_grad()
                loss = loss_func(pred, label)
                loss.backward()
                optimizer.step()
                experiment.log_metric("loss", loss.detach().cpu())


def test(model, test_loader, experiment, hyperparams, bpe):
    """
    Validates the model performance as LM on never-seen data.

    :param model: the trained model to use for prediction
    :param test_loader: Dataloader of testing data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    :param bpe: is bpe dataset or not
    """
    # TODO: Define loss function, total loss, and total word count
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    total_loss = 0
    word_count = 0

    total_pred = 0
    correct_pred = 0

    model = model.eval()
    with experiment.test():
        # TODO: Write testing loop
        for batch in tqdm(test_loader):
            enc_data = batch["enc_data"].to(device)
            dec_data = batch["dec_data"].to(device)
            label = batch["label"].to(device)
            enc_length = batch["enc_length"].to(device)
            dec_length = batch["dec_length"].to(device)

            logits = model(enc_data, dec_data, enc_length, dec_length).detach()
            pred = torch.flatten(logits, start_dim=0, end_dim=1)
            label = torch.flatten(label)

            loss = loss_fn(pred, label)
            total_loss += loss
            word_count += torch.sum(enc_length)

            mask = label != 0
            assert(mask.shape == label.shape)
            pred_idx = torch.argmax(pred, 1)
            correct_pred += torch.sum(mask * (pred_idx == label))
            total_pred += torch.sum(mask)

        perplexity = torch.exp(total_loss / word_count)
        accuracy = int(correct_pred) / int(total_pred)
        print("perplexity:", perplexity)
        print("accuracy: {}/{}={}".format(correct_pred, total_pred, accuracy))
        experiment.log_metric("perplexity", perplexity.cpu())
        experiment.log_metric("accuracy", accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--corpus-files", nargs="*")
    parser.add_argument("-b", "--bpe", action="store_true",
                        help="use bpe data")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-m", "--multilingual-tags", nargs="*", default=[None],
                        help="target tags for translation")
    parser.add_argument("-z", "--zeroshot", action="store_true",
                        help="zeroshot translation")
    args = parser.parse_args()

    # TODO: Make sure you modify the `.comet.config` file
    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparams)

    # TODO: Load dataset
    # Hint: Use random_split to split dataset into train and validate datasets
    # Hint: Use ConcatDataset to concatenate datasets
    # Hint: Make sure encoding and decoding lengths match for the datasets
    data_tags = list(zip(args.corpus_files, args.multilingual_tags))

    data_tags = list(zip(args.corpus_files, args.multilingual_tags))
    if len(data_tags) == 1:
        dataset = TranslationDataset(data_tags[0][0], None, None, bpe=args.bpe)
        if args.bpe:
            vocab_size, output_size = len(dataset.word2id), len(dataset.word2id)
        else:
            vocab_size, output_size = len(dataset.src_word2id), len(dataset.eng_word2id)

    else:
        enc_seq_len, dec_seq_len = 0, 0
        for corpus_file, _ in data_tags:
            target, source = read_from_corpus(corpus_file)
            enc_seq_len = max(enc_seq_len, max([len(line) for line in source]) + 2)
            dec_seq_len = max(dec_seq_len, max([len(line) for line in source]) + 1)

        sets = list()
        word2id = None
        for corpus_file, tag in data_tags:
            dataset = TranslationDataset(corpus_file, enc_seq_len, dec_seq_len, bpe=args.bpe, target=tag, word2id=word2id)
            sets.append(dataset)
            word2id = dataset.word2id
        vocab_size, output_size = len(word2id), len(word2id)
        dataset = ConcatDataset(sets)

    
    test_size = len(dataset) // 10
    train_size = len(dataset) - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=hyperparams["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=hyperparams["batch_size"], shuffle=True)

    print("vocab size", vocab_size, output_size)
    model = Seq2Seq(
        # TODO: Fill this initiator
        vocab_size,
        hyperparams["rnn_size"],
        hyperparams["embedding_size"],
        output_size,
        None,
        None,
        args.bpe
    ).to(device)

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        print("running training loop...")
        train(model, train_loader, experiment, hyperparams, args.bpe)
    if args.test:
        print("running testing loop...")
        test(model, test_loader, experiment, hyperparams, args.bpe)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
