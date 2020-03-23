from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Looking at the current dataset
dataset = pd.read_csv("plants.data") # Not sure what to do with the error
print(dataset.head())