import torch
import torch.nn as nn
import torch.nn.functional as F

class POSModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):

        super(POSModel, self).__init__()

        # remove ?! self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # initialize hidden layer
        self.hidden = self.init_hidden()

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # prepares output from lstm for softmax classifier,
        # maps from (seq.length x hdden_dim) to (seq.length x target_size)
        self.dense = nn.Linear(hidden_dim, target_size)


    # from the pytorch tutorial
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, word_indices):
        seq_length = len(word_indices)
        word_embeddings = self.embeddings(word_indices)
        # word embeddings is seq.length x n shape, transfrom it to seq.length x 1 x n
        # with n is ???? hidden_dim
        lstm_input = word_embeddings.view(seq_length,1,-1)
        output, self.hidden = self.lstm(lstm_input, self.hidden)

        #flatten output
        output = output.view(seq_length, -1)
        output = self.dense(output)
        return F.log_softmax(output, dim=1)
