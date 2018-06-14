import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import treebank
from model import POSModel


def create_word_idx(words):
    word_to_index = {k: v for v, k in enumerate(set(words))}
    return word_to_index

def create_tag_idx(sentences):
    tags = {t for sent in sentences for (_,t) in sent}
    tag_to_index = {k: t for t, k in enumerate(tags)}
    return tag_to_index


def convertToVec(sentence, word_idx, tag_idx):
    wordvec = []
    tagvec = []
    for w, t in sentence:
        wordvec.append(word_idx[w]), tagvec.append(tag_idx[t])

    return torch.tensor(wordvec), torch.tensor(tagvec)

def train(model, word_to_index, tag_to_index, training_data, epochs):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    for epoch in range(epochs):

        lossValue = 0.
        for tagged_sentence in training_data:

            # pytroch accumlates things, do some housekeeping here
            model.zero_grad() # clear gradients
            model.hidden = model.init_hidden() # clear history of the lstm for this training instance

            x_data, y_data = convertToVec(tagged_sentence, word_to_index, tag_to_index)

            # forward pass
            result = model(x_data)

            #calculate loss,
            loss = criterion(result, y_data)
            lossValue = loss.item()

            # compute gradients
            loss.backward()

            # update gradients
            optimizer.step()
        print(lossValue)


sentences = treebank.tagged_sents()
words = treebank.words()

word_to_index = create_word_idx(words)
tag_to_index = create_tag_idx(sentences)
hidden_dim = 32
embedding_dim = 64

training_data = sentences[:3000]
test_data = sentences[3000:]
print("len sentences: ", len(sentences), "training: ", len(training_data), "test: ", len(test_data))

vocab_size = len(word_to_index)
target_size = len(tag_to_index)

model = POSModel(embedding_dim, hidden_dim, vocab_size, target_size)

train(model, word_to_index, tag_to_index, training_data, 30)

#scores after training
with torch.no_grad():
    x, y = convertToVec(test_data[0], word_to_index, tag_to_index)
    output = model(x)
    _, pred = torch.max(output, 1)
    print("pred: ", pred)
    print("target: ", y)

