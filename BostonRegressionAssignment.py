import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Looking at the current dataset
# Looking at the dataset, we can see that we have three arrays: data, target and feature_names
data = load_boston()
print(data)
