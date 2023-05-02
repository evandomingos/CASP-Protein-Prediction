# Various Libraries, some possibly redundant
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale


dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv", header=0)

protein = dataset.dropna()

