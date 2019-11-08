Dataset can be downloaded from - 
https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews. 
https://drive.google.com/file/d/1iCxHFV6Y383pIFt6on8a0ygRRhO0oipH/view?usp=sharing
Original dataset can be downloaded from - http://ai.stanford.edu/~amaas/data/sentiment/


* data_exp.py - For identifying the max length of the sentence.
* data_prep.py - Loads the dataset and splits into train and test set.
* preprocess.py - Tokenize and apply GloVe embedding on the dataset.
* models/ - model files.
* main.py - model training and evaluation.

LSTM Parameters ( Batch size = 512, epochs = 4  for Test accuracy = 0.804)
LSTM with Attention Parameters ( Batch size = 512, epochs = 4 for Test accuracy = 0.704)