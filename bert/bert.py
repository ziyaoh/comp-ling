from comet_ml import Experiment
from data import MyDataset, read_file
from model import BERT
from embedding_analysis import embedding_analysis
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar

# TODO: Set hyperparameters
hyperparams = {
    "num_epochs": 80,
    "batch_size": 32,
    "lr": 0.0001,
    "seq_len": 100,

    "hidden_size": 512,
    "num_head": 4,
    "num_layers": 2,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, loss_fn, optimizer, experiment, hyperparams, test_loader):
    """
    Training loop that trains BERT model.

    Inputs:
    - model: A BERT model
    - train_loader: Dataloader of training data
    - experiment: comet.ml experiment object
    - hyperparams: Hyperparameters dictionary
    """
    model = model.train()
    with experiment.train():
        # TODO: Write training loop
        for epoch in range(hyperparams["num_epochs"]):
            print("training for epoch", epoch)
            for batch in tqdm(train_loader):
                data = batch["data"].to(device)
                label = batch["label"].to(device)
                masked_ind = batch["masked_ind"].to(device)

                # Forward + Backward + Optimize
                logits = model(data, masked_ind)
                pred = torch.flatten(logits, start_dim=0, end_dim=1)
                label = torch.flatten(label)
                loss = loss_fn(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(model.embedding.weight.grad)
                # input("check grad")
                experiment.log_metric("loss", loss.detach().cpu())
            """
            if (epoch + 1) % 10 == 0:
                test(model, test_loader, loss_fn, experiment, hyperparams)
            if (epoch + 1) % 20 == 0:
                embedding_analysis(model, experiment, train_loader.dataset, test_loader.dataset, epoch + 1)
            """


def test(model, test_loader, loss_fn, experiment, hyperparams):
    """
    Testing loop for BERT model and logs perplexity and accuracy to comet.ml.

    Inputs:
    - model: A BERT model
    - train_loader: Dataloader of training data
    - experiment: comet.ml experiment object
    - hyperparams: Hyperparameters dictionary
    """
    model = model.eval()
    total_loss = 0
    word_count = 0
    correct_count = 0
    with experiment.test(), torch.no_grad():
        # TODO: Write testing loop
        for batch in tqdm(test_loader):
            data = batch["data"].to(device)
            label = batch["label"].to(device)
            masked_ind = batch["masked_ind"].to(device)

            logits = model(data, masked_ind)

            pred = torch.flatten(logits, start_dim=0, end_dim=1)
            label = torch.flatten(label)

            pred_idx = torch.argmax(pred, 1)
            correct_count += torch.sum(pred_idx == label)

            loss = loss_fn(pred, label)

            num = label.shape[0]
            total_loss += (loss * num)
            word_count += num

        print("total_loss", total_loss, "word_count", word_count, "correct_count", correct_count)
        perplexity = torch.exp(total_loss / word_count).cpu()
        accuracy = int(correct_count) / int(word_count)
        experiment.log_metric("perplexity", perplexity)
        experiment.log_metric("accuracy", accuracy)
        print("perplexity:", perplexity)
        print("accuracy:", accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("test_file")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-a", "--analysis", action="store_true",
                        help="run embedding analysis")
    args = parser.parse_args()

    # TODO: Make sure you modify the `.comet.config` file
    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparams)

    # TODO: Load dataset
    train_set = MyDataset(args.train_file, hyperparams["seq_len"])
    word2id= train_set.word2id
    test_set = MyDataset(args.test_file, hyperparams["seq_len"], word2id)
    train_loader = DataLoader(train_set, batch_size=hyperparams["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=hyperparams["batch_size"], shuffle=False)
    num_tokens = len(word2id)

    model = BERT(num_tokens, hidden_size=hyperparams["hidden_size"], num_head=hyperparams["num_head"], num_layers=hyperparams["num_layers"]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])

    if args.load:
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        train(model, train_loader, loss_fn, optimizer, experiment, hyperparams, test_loader)
    if args.test:
        test(model, test_loader, loss_fn, experiment, hyperparams)
    if args.save:
        torch.save(model.state_dict(), './model.pt')
    if args.analysis:
        embedding_analysis(model, experiment, train_set, test_set)
