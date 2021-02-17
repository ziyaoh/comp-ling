from comet_ml import Experiment
from preprocess import ParsingDataset, RerankingDataset
from model import LSTMLM
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
import argparse
import random
import os
from tqdm import tqdm  # optional progress bar

# TODO: Set hyperparameters
hyperparams = {
    "rnn_size": 64,
    "embedding_size": 64,
    "num_epochs": 1,
    "batch_size": 32,
    "learning_rate": 0.01
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, experiment, hyperparams):
    """
    Trains the model.

    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    """
    # TODO: Define loss function and optimizer

    # TODO: Write training loop
    model = model.train()
    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    with experiment.train():
        for epoch in range(hyperparams["num_epochs"]):
            print("training for epoch", epoch)
            for batch in tqdm(train_loader):
                data = batch["data"].to(device)
                label = batch["label"].to(device)
                length = batch["length"].to(device)

                logits = model(data, length)
                pred = torch.flatten(logits, start_dim=0, end_dim=1)
                label = torch.flatten(label)

                optimizer.zero_grad()
                loss = loss_func(pred, label)
                loss.backward()
                optimizer.step()
                experiment.log_metric("loss", loss.detach())


def validate(model, validate_loader, experiment, hyperparams):
    """
    Validates the model performance as LM on never-seen data.

    :param model: the trained model to use for prediction
    :param validate_loader: Dataloader of validation data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    """
    # TODO: Define loss function, total loss, and total word count
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    total_loss = 0
    word_count = 0

    # TODO: Write validating loop
    model = model.eval()
    with experiment.validate():
        for batch in tqdm(validate_loader):
            data = batch["data"].to(device)
            label = batch["label"].to(device)
            length = batch["length"].to(device)

            logits = model(data, length)
            pred = torch.flatten(logits, start_dim=0, end_dim=1)
            label = torch.flatten(label)

            loss = loss_fn(pred, label).detach()
            total_loss += loss
            word_count += torch.sum(length)

        perplexity = torch.exp(total_loss / word_count)
        print("perplexity:", perplexity)
        experiment.log_metric("perplexity", perplexity)


def test(model, test_dataset, experiment, hyperparams):
    """
    Validates and tests the model for parse reranking.

    :param model: the trained model to use for prediction
    :param test_dataset: Dataloader of testing data
    :param experiment: comet.ml experiment object
    :param hyperparams: Hyperparameters dictionary
    """
    # TODO: Write testing loops
    model = model.eval()
    with experiment.test():
        num_correct, num_parsed, num_gold = 0, 0, 0
        for batch in tqdm(test_dataset):
            data = batch["datas"].to(device)
            label = batch["labels"].to(device)
            length = batch["lengths"].to(device)
            parsing_res = batch["parsing_res"]

            logits = model(data, length).detach()
            prob = F.softmax(logits, dim=-1)

            selected_prob = torch.gather(input=prob, dim=2, index=label.unsqueeze(2)).squeeze(2)
            mask = (label != 0) * torch.ones([label.shape[0], label.shape[1]]).to(device)
            final_prob = torch.sum((- torch.log(selected_prob) * mask), 1)
            chosen = torch.argmax(-final_prob)

            ########################### randome baseline ###########################
            # chosen = random.randint(0, len(parsing_res) - 1)

            num_correct += parsing_res[chosen][0]
            num_parsed += parsing_res[chosen][1]
            num_gold += batch["num_gold"]

        precision = num_correct / num_parsed
        recall = num_correct / num_gold
        f1 = 2 * precision * recall / (precision + recall)
        print("precision:", precision)
        print("recall:", recall)
        print("F1:", f1)
        experiment.log_metric("precision", precision)
        experiment.log_metric("recall", recall)
        experiment.log_metric("F1", f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("parse_file")
    parser.add_argument("gold_file")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-v", "--validate", action="store_true",
                        help="run validation loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    args = parser.parse_args()


    # TODO: Load dataset
    # Hint: Use random_split to split dataset into train and validate datasets
    print("loading dataset...")
    train_dataset = ParsingDataset(args.train_file)
    validate_size = len(train_dataset) // 10
    train_size = len(train_dataset) - validate_size
    train_set, validate_set = random_split(train_dataset, [train_size, validate_size])
    train_loader = DataLoader(train_set, batch_size=hyperparams["batch_size"], shuffle=True)
    validate_loader = DataLoader(validate_set, batch_size=hyperparams["batch_size"], shuffle=True)

    test_dataset = RerankingDataset(args.parse_file, args.gold_file, train_dataset.word2id)

    vocab_size = len(train_dataset.word2id)
    model = LSTMLM(
        vocab_size,
        hyperparams["rnn_size"],
        hyperparams["embedding_size"]
    ).to(device)

    # TODO: Make sure you modify the `.comet.config` file
    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparams)

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        print("running training loop...")
        train(model, train_loader, experiment, hyperparams)
    if args.validate:
        print("running validation...")
        validate(model, validate_loader, experiment, hyperparams)
    if args.test:
        print("testing reranker...")
        test(model, test_dataset, experiment, hyperparams)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')

    experiment.end()
