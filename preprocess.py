import pandas as pd 
import nltk
import torchtext
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
#nltk.download('punkt')

def preproc_load(trainpath : str, testpath :str, min_freq :int=1, max_sent_length:int=256,test_sen= None, save="models/word2index.pkl" ):
    text = data.Field(sequential = True, use_vocab=True, tokenize=nltk.word_tokenize, lower=True, init_token="</bos>", eos_token="</eos>", include_lengths=False, is_target=False, batch_first=True, fix_length=max_sent_length )
    target = data.LabelField(sequential=False, use_vocab=False, batch_first=True, is_target=True )
    dataset = torchtext.data.TabularDataset(trainpath, format="csv", fields={"target": ('target',target), "text":('text',text)})
    data_test = torchtext.data.TabularDataset(testpath, format="csv", fields ={"target": ('target', target), "text":('text', text)})

    #vocab building
    text.build_vocab(dataset, data_test,min_freq=min_freq)
    text.vocab.load_vectors(torchtext.vocab.GloVe(name="6B", dim=100))
    vocab_size = len(text.vocab.itos)
    padding_idx = text.vocab.stoi[text.pad_token]

    #split dataset to train and valid
    data_train, data_valid = dataset.split(split_ratio=0.8)

    print("trainset size", len(data_train))
    print("validset size", len(data_valid))
    print("testset size", len(data_test))
    print("Vocabsize", vocab_size)
    print("embed shape", text.vocab.vectors.shape)
    argsdict = {
        "data_train" : data_train,
        "data_valid" : data_valid,
        "data_test" : data_test,
        "vocab_size" : vocab_size,
        "pretrained_embeddings" : text.vocab.vectors,
        "padding_idx" : padding_idx }
    return argsdict

#data_train,data_valid,data_test,pretrained_embeddings,padding_idx = preproc_load(file_train,file_test)