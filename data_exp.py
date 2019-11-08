import pandas as pd 
import matplotlib.pyplot as plt 

data_file = "data/dataset_orig.csv"

df = pd.read_csv(data_file, encoding='latin-1')
df["length"] = df["review"].str.split().str.len()


fig=plt.figure(figsize=(100,10))
df.hist(column="length")
plt.show()
print(df["length"].describe())
