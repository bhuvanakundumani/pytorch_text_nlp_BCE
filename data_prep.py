import pandas as pd
from sklearn import preprocessing
import collections
from sklearn.model_selection import train_test_split

data_file = "data/dataset_orig.csv"

def load_dataset(data_file):
    '''
    Reads and converts the original dataset to format("target","text").
    shuffles the dataset and stores it to "imdb_dataset.csv")
    Label encode the target columns.
    '''
    df = pd.read_csv(data_file, encoding='latin-1')
    df.columns = ['text', 'target']
    df = df[["target","text"]]
    # To load the entire dataset set frac=1
    df = df.sample(frac=0.1, random_state=42)
    df.to_csv("data/imdb_dataset.csv", index=False)
    dataf = pd.read_csv("data/imdb_dataset.csv")
    labelencoder = preprocessing.LabelEncoder()
    dataf["target"] = labelencoder.fit_transform(dataf["target"])
    target_map = {k:i for k,i in enumerate(labelencoder.classes_)}
    print(target_map)
    c = collections.Counter(dataf["target"])
    print(c)
    return dataf

def split_dataset(df):
    '''
    splits the dataset into train and test 
    '''
    train, test = train_test_split(df, test_size=0.15, random_state=42)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    df_train = pd.read_csv("data/train.csv")
    print(df_train.shape)
    df_test = pd.read_csv("data/test.csv")
    print(df_test.shape)
   
dataf = load_dataset(data_file)
split_dataset(dataf)

    