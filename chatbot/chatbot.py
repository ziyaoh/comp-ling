from comet_ml import Experiment
import torch
import torch.nn
import argparse
import math
import numpy as np
from preprocess import *
from tqdm import tqdm

from transformers import GPT2Tokenizer, GPT2LMHeadModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyper_params = {
     "batch_size": None,
     "num_epochs": None,
     "learning_rate": None,
     "window_size": None
 }


def train(model, train_loader, optimizer, experiment):
    """
    Trains the model.
    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param optimizer: the initilized optimizer
    :param experiment: comet.ml experiment object
    """
    # TODO: Write the training loop here, save trained model weights if needed
    model = model.train()
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)
    with experiment.train():
        for epoch in range(hyper_params["num_epochs"]):
            print("training for epoch", epoch)
            for batch in tqdm(train_loader):
                data = batch["data"].to(DEVICE)
                label = batch["label"].to(DEVICE)
                # Forward + Backward + Optimize
                logits = model(input_ids=data)
                pred = torch.flatten(logits, start_dim=0, end_dim=1)
                label = torch.flatten(label)

                optimizer.zero_grad()
                loss = loss_func(pred, label)
                loss.backward()
                optimizer.step()
                experiment.log_metric("loss", loss.detach().cpu())


def test(model, test_loader, experiment):
    """
    Validates the model performance as LM on never-seen data using perplexity.
    :param model: the trained model to use for testing
    :param test_loader: Dataloader of testing data
    :param experiment: comet.ml experiment object
    """
    # TODO: Write the testing loop and calculate perplexity
    total_loss = 0
    word_count = 0
    model = model.eval()
    with experiment.validate():
        with torch.no_grad():
            for batch in tqdm(test_loader):
                data = batch["data"].to(DEVICE)
                label = batch["label"].to(DEVICE)
                length = batch["length"].to(DEVICE)

                loss = model(input_ids=data, labels=label)
                num = torch.sum(length)
                # print(loss)
                # print(num)
                total_loss += (loss * num)
                word_count += num
        # Log perplexity to Comet.ml using experiment.log_metric
        perplexity = torch.exp(total_loss / word_count)
        print("perplexity:", perplexity)
        experiment.log_metric("perplexity", perplexity.cpu())


def interactive(input, tokenizer, model, top_k=10, ntok=20):
    """
    Generate and print out the response given input using the trained model
    :param input: an input string as prompt (i.e. How are you?)
    :param tokenizer: intialized tokenizer object for encoding the input
    :param model: the trained model to use for generate prediction
    :param top_k: number of samples for top_l sampling
    :param ntok: maximum number of tokens to generate

    Comment: Feed in the input to the model to generate the most probable token
    and concatenate it with current input.
    Continue this process iteratively until the model predicts the padding
    token or reach the maximum number of tokens.
    You may need to add the BOS token and special token to the input sentence
    before passing into model.
    Also, you may want to filter out your input sentence and meaningless tokens
    when printing out the response.
    """
    # TODO: Write the generation function for interacting with trained model
    prompt_ids = tokenizer.encode(input)
    data = torch.LongTensor([tokenizer.bos_token_id] + prompt_ids + [tokenizer.sep_token_id]).to(DEVICE)

    result = model.generate(input_ids=data, max_length=ntok, do_sample=True, top_k=top_k, eos_token_id=tokenizer.eos_token_id, forced_eos_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(result[0][data.shape[0]: -1])
    print(response)


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
    parser.add_argument("-i", "--interative", action="store_true",
                        help="run in interactive mode")
    args = parser.parse_args()

    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyper_params)

    # Load the GPT2 Tokenizer, add any special token if needed
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Intialized the pretrained GPT-2 model and optimizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    optimizer = torch.optim.Adam(model.parameters(), lr = hyper_params['learning_rate'])

    # Load the train, test DataLoader NOTE: Parse the data using GPT2 tokenizer
    train_loader, test_loader = load_dataset((args.train_file, args.test_file), tokenizer, hyper_params["batch_size"])

    if args.load:
        model.load_state_dict(torch.load('model.pt'))
    if args.train:
        # run train loop here
        print("running training loop...")
        train(model, train_loader, optimizer, experiment)
    if args.save:
        torch.save(model.state_dict(), 'model.pt')
    if args.test:
        # run test loop here
        print("running testing loop...")
        test(model, test_loader, experiment)
    if args.interactive:
        # generate your own chat with the model here
        print("running interative mode...")
        while True:
            input_text = input("Please say something: ")
            interactive(input_text, tokenizer, model)
