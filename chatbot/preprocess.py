from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

SEP = "<|septoken|>"

class ChatbotDataset(Dataset):

    def __init__(self, file_path, tokenizer):
        bos, sep, eos = tokenizer.bos_token_id, tokenizer.sep_token_id, tokenizer.eos_token_id
        # print(bos, tokenizer.bos_token, sep, tokenizer.sep_token, eos, tokenizer.eos_token)

        with open(file_path, "r") as f:
            sentences = f.readlines()

        sentences = [sent.strip() for sent in sentences]

        self.all_data = list()
        self.all_label = list()
        self.all_length = list()
        for sentence in sentences:
            pair = sentence.split("\t")
            if len(pair) != 2:
                continue
            prompt, response = pair
            prompt = prompt.strip()
            response = response.strip()
            prompt = " ".join(prompt.split()[1:])

            prompt_id = tokenizer.encode(prompt)
            response_id = tokenizer.encode(response)

            data = prompt_id + [sep] + response_id
            label = [-100] * (len(prompt_id)) + response_id + [eos]
            assert(len(data) == len(label))
            self.all_data.append(torch.LongTensor(data))
            self.all_label.append(torch.LongTensor(label))
            self.all_length.append(len(response_id) + 1)


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return {
            "data": self.all_data[idx],
            "label": self.all_label[idx],
            "length": self.all_length[idx],
        }


class MyCollator:
    def __init__(self, data_padding_value, label_padding_value):
        self.data_padding_value = data_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, batch):
        data, label, length, mask = list(), list(), list(), list()
        for item in batch:
            data.append(item["data"])
            label.append(item["label"])
            length.append(item["length"])
            mask.append(torch.FloatTensor([1] * len(item["data"])))
        data = pad_sequence(data, batch_first=True, padding_value=self.data_padding_value)
        label = pad_sequence(label, batch_first=True, padding_value=self.label_padding_value)
        mask = pad_sequence(mask, batch_first=True, padding_value=0)
        position = torch.LongTensor([[list(range(data.shape[1]))] for _ in range(data.shape[0])])
        return {
            "data": data,
            "label": label,
            "length": torch.LongTensor(length),
            "mask": mask,
            "position": position,
        }


def load_dataset(fn, tokenizer, batch_size):
    """
    :param fn: filename for the dataset
    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for
    train and test

    :Comment: This function should be similary as the last GPT-2 assignemnent.
    We are still using the GPT-2 tokenizer to get the word ids.
    One thing to note is that when preprocessing the dataset, please exclude
    all the sentences defining the speaker's persona, in this assignment,
    we will just be training and testing on the chat text data. Also for each
    record of chat in the format of "sentence \t response", add BOS and EOS
    token to the start and end it, also add a special token between sentence
    and response to differentiate them from each other.
    """
    train_set = ChatbotDataset(fn[0], tokenizer)
    test_set = ChatbotDataset(fn[1], tokenizer)
    
    train_collator = MyCollator(0, -100)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=train_collator)

    test_collator = MyCollator(0, -100)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=test_collator)

    return train_loader, test_loader
