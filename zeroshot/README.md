## Zeroshot

Commet.ml: https://www.comet.ml/fxgtqr5z/zeroshot/view/new

### Result

#### BPE English -> French
* comet hash: 4ac6953a91454b7f8634e8cf42d2a8cd
* perplexity: 5.02
* accuracy: 71.8%

#### BPE English -> German
* comet hash: f88a9c7cd39b48878ad6f8bc118fbc77
* perplexity: 5.35
* accuracy: 71.2%


#### Joint BPE English -> French/German
* comet hash: 070bf6fd0b6e42859024492c362d0143
* perplexity: 1.54
* accuracy:  72.8%

#### Zeroshot German -> French
* comet hash: 1b4b53421c294da690f92f05dd6111a4
* perplexity: 262.67
* accuracy: 28.5%

**Zeroshot usage**

To run zeroshot experiment, turn on the zeroshot flag, and pass in paths to the four data files in EXACTLY the following order. Don't put in the -m flag.

```
python zeroshot.py -Ttbz -f [train eng deu file] [train eng fra file] [test eng deu file] [test eng fra file]
```

### Hyperparameters and Model

hyperparams = {
    "rnn_size": 256,
    "embedding_size": 128,
    "num_epochs": 2,
    "batch_size": 64,
    "learning_rate": 0.001, 
}

My model uses two unidirectional GRU layers, one for encoder and the other for decoder. I use GRU instead of LSTM because they achieve similar performance while GRU is slightly simpler for coding purpose. I also utilize attention mechanish to help construct context for decoder.

I choose rnn size and embedding size to be 256 and 128 out of several experiments. This combination achieves comparable performance against larger models, while being a lot faster to train. The model and hyperparameters are mostly identical to my multilingual project, with one difference being the number epochs changed from 3 to 2. This is mostly for training time concerning.

### Discussion

My one to many model happens to perform slightly better than my two one to one models. I think the reason migh be that the data I used for training the two one to one models went through the BPE processing jointly. This could lower the model's capability to capture the features behind the data.

I was able to replicate the zeroshot translation. Even though the model has never seen the data from German to French, the encoder and decoder are doing things they have learned. Namely encode German to context vector and decode context vector to French. Therefore, the model was able to more or less translate German to French, despite the low accuracy.