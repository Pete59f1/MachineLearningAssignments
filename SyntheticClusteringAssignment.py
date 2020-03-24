import pandas as pd

# Looking at the current dataset
data = pd.read_csv("s1-cb.txt", header=None)
print(data.head())