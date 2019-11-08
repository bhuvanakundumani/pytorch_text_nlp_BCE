import pandas as pd 
import numpy as np
import torch
import torchtext
import nltk
from tqdm import tqdm
from models.BCELogits.LSTM import LstmClassifier
from models.BCELogits.LSTM_Attn import LstmAttentionModel
from preprocess import preproc_load
from sklearn.metrics import accuracy_score
import torch.nn as nn

file_train = "data/train.csv"
file_test ="data/test.csv"

data_dict = preproc_load(file_train, file_test)
train_data = data_dict["data_train"]
test_data = data_dict["data_test"]
valid_data = data_dict["data_valid"]
epochs = 4
lr = 0.01

output_size = 1
hidden_size = 100
embedding_length = 100
batch_size = 512
#model = LstmClassifier(batch_size, output_size, hidden_size, data_dict["vocab_size"], embedding_length, data_dict["pretrained_embeddings"])
model = LstmAttentionModel(batch_size, output_size, hidden_size, data_dict["vocab_size"], embedding_length, data_dict["pretrained_embeddings"])
optimizer = torch.optim.Adam(model.parameters(), lr)
loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')

def train_model(model, train_iter, optimizer):
    train_epoch_loss = 0
    train_true = []
    train_pred = []
    optimizer = optimizer
    steps = 0
    model.train()

    for idx, batch in enumerate(train_iter):
        text, target = batch.text, batch.target 
        optimizer.zero_grad()
        target = target.float()
        output = model(text)
        loss = loss_function(output, target.view(-1,1))
        loss.backward()
        optimizer.step()
        train_epoch_loss  += loss.item()

        pred = np.round(torch.sigmoid(output).detach().numpy())
        train_pred.extend(pred.reshape(-1,1).tolist())
        train_true.extend(target.float().tolist())

    accuracy_train = accuracy_score(train_true, train_pred)
    return train_epoch_loss, accuracy_train

def valid_model(model, valid_iter):
    valid_epoch_loss = 0
    valid_true = []
    valid_pred = []
    model.eval()
    for idx,batch in enumerate(valid_iter):
        text = batch.text
        target = batch.target
        target = target.float()
        with torch.no_grad():
            valid_true.extend(target.float().tolist())
            output = model(text)
            loss = loss_function(output, target.view(-1,1))
            valid_epoch_loss  += loss.item()
            pred = np.round(torch.sigmoid(output).detach().numpy())
            valid_pred.extend(pred.reshape(-1,1).tolist())

    accuracy_valid = accuracy_score(valid_true,valid_pred)
    return valid_epoch_loss, accuracy_valid

def test_model(model, test_iter):
    test_epoch_loss = 0
    test_true = []
    test_pred = []
    model.eval()
    for idx,batch in enumerate(test_iter):
        text = batch.text
        target = batch.target
        target = target.float()
        with torch.no_grad():
            test_true.extend(target.float().tolist())
            output = model(text)
            loss = loss_function(output, target.view(-1,1))
            test_epoch_loss += loss.item()
            pred = np.round(torch.sigmoid(output).detach().numpy())
            test_pred.extend(pred.reshape(-1,1).tolist())

    accuracy_test = accuracy_score(test_true,test_pred)
    return test_epoch_loss, accuracy_test

train_iter = torchtext.data.Iterator(dataset=train_data, batch_size=batch_size, train=True, shuffle=True, sort=False)
valid_iter = torchtext.data.Iterator(dataset=valid_data, batch_size=batch_size, train=True, shuffle=True, sort=False)
test_iter = torchtext.data.Iterator(dataset=test_data, batch_size=batch_size, train=True, shuffle=True, sort=False)

for epoch in range(epochs):
    print("********************************************************")
    print(f"epoch number {epoch}")
    train_loss, train_accu = train_model(model, train_iter,optimizer)
    valid_loss, valid_accu = valid_model(model, valid_iter)
    print(f"Train loss:{train_loss:.3f}, Train accu:{train_accu:.3f}, Valid loss:{valid_loss:.3f}, Valid accu:{valid_accu:.3f}")

test_loss, test_accu = test_model(model, test_iter)
print("___________________________________________________")
print(f"Test Loss:{test_loss:.3f}, Test acc:{test_accu:.3f}")

test_sentence = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues"

