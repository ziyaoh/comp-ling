## Multilingual

Commet.ml: https://www.comet.ml/fxgtqr5z/multilingual/view/new

### Result

#### BPE French -> English
* comet hash: 4603d82fb08949ab99404deec85397bb
* perplexity: 2.90
* accuracy: 77.7%

#### Vanila French -> English
* comet hash: 47ee49390c2a4656af5541c86dfb8973
* perplexity: 3.88
* accuracy: 70.8%

#### Joint BPE French/German -> English
* comet hash: ef7ad699e0694d84a3d961fdbdccafac
* perplexity: 1.32
* accuracy: 79.3%

### Hyperparameters and Model

hyperparams = {
    "rnn_size": 256,
    "embedding_size": 128,
    "num_epochs": 3,
    "batch_size": 64,
    "learning_rate": 0.001, 
}

My model uses two unidirectional GRU layers, one for encoder and the other for decoder. I use GRU instead of LSTM because they achieve similar performance while GRU is slightly simpler for coding purpose. I also utilize attention mechanish to help construct context for decoder.

I choose rnn size and embedding size to be 256 and 128 out of several experiments. This combination achieves comparable performance against larger models, while being a lot faster to train.

### BPE vs Traditional Methods

BPE breaks words into common sub-words, and thereby removes the necessaity of UNK uncommon words. On the other hand, the BPE process relies on the word frequency, which really depends on the corpus. So the performance of the same model might be unstable if the corpus changes.