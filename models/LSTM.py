import torch
import torch.nn as nn
import torch.nn.functional as functional

class LstmClassifier(nn.Module):
    def __init__(self,batch_size,output_size,hidden_size,vocab_size,embedding_length, pretrained_embed):
        super(LstmClassifier,self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.embed = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)
        self.lstm = nn.LSTM(embedding_length,hidden_size)
        self.linear = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.output = nn.Linear(16,output_size)


    def forward(self, x):
        h_embedding = self.embed(x)
        output, _ = self.lstm(h_embedding)
        maxpool, _ = torch.max(output, 1)
        linear = self.relu(self.linear(maxpool))
        out = self.output(linear)
        return out