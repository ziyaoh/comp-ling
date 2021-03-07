from comet_ml import Experiment
import torch
from torch import nn, optim
import argparse
from transformers import *
from gpt2 import *
from transformer import *
from model import *
from preprocess import *
from tqdm import tqdm


hyper_params = {
    "batch_size": 100,
    "num_epochs": 3,
    "learning_rate": 0.01,

    "hidden_size": 128,
    "embedding_size": 64,
    "num_head": 4,
    "dropout_rate": 0.1,
    "num_layer": 2,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the Model
def train(model, train_loader, experiment, hyperparams):
    model = model.train()
    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    with experiment.train():
        for epoch in range(hyperparams["num_epochs"]):
            print("training for epoch", epoch)
            for batch in tqdm(train_loader):
                data = batch["data"].to(device)
                label = batch["label"].to(device)
                # Forward + Backward + Optimize
                logits = model(data)
                pred = torch.flatten(logits, start_dim=0, end_dim=1)
                label = torch.flatten(label)

                optimizer.zero_grad()
                loss = loss_func(pred, label)
                loss.backward()
                optimizer.step()
                experiment.log_metric("loss", loss.detach())

                # Compute train accuracy

                # Log perplexity to Comet.ml using experiment.log_metric



# Test the Model
def test(model, test_loader, experiment, hyperparams):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    total_loss = 0
    word_count = 0

    model = model.eval()
    with experiment.test():
        for batch in tqdm(test_loader):
            data = batch["data"].to(device)
            label = batch["label"].to(device)
            length = batch["length"].to(device)
            logits = model(data)

            pred = torch.flatten(logits, start_dim=0, end_dim=1)
            label = torch.flatten(label)
            loss = loss_fn(pred, label).detach()
            total_loss += loss
            word_count += torch.sum(length)
        # Log perplexity to Comet.ml using experiment.log_metric
        perplexity = torch.exp(total_loss / word_count)
        print("perplexity:", perplexity)
        experiment.log_metric("perplexity", perplexity)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("test_file")
    parser.add_argument("-m", "--model", type=str, default="",
                        help="transformer or gpt2")
    parser.add_argument("-bs", "--batch_size", type=int, default=1,
                        help="batch size")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    args = parser.parse_args()

    experiment = Experiment(project_name="gpt2")
    experiment.log_parameters(hyper_params)

    # Load the GPT2 Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if args.model == "transformer":
        tokenizer.add_tokens(["START", "STOP"])
   
    # Load the train, test DataLoader NOTE: Parse the data using the GPT2 tokenizer for both models
    train_loader, test_loader, vocab_size = load_dataset((args.train_file, args.test_file), tokenizer, args.model, args.batch_size)

    if args.model == "transformer":
        # Load your transformer
        model = Transformer(
            vocab_size,
            hyper_params["hidden_size"],
            hyper_params["num_head"],
            hyper_params["num_layer"],
            hyper_params["dropout_rate"]
        )
    elif args.model == "gpt2":
        # Load the GPT2 model
        model = GPT2_Transformer()

    # Train the model if args.model == "transformer"


    # Test the model on the test set - report perplexity

    if args.train:
        if args.model == "transformer":
            print("running training loop...")
            train(model, train_loader, experiment, hyper_params)
    if args.test:
        print("testing reranker...")
        test(model, test_loader, experiment, hyper_params)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')

    experiment.end()
