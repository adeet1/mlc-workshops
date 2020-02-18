import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("spam.csv", encoding = "ISO-8859-1")

df.rename(columns = {"v1": "label",
                     "v2": "message"}, inplace = True) # this does the same thing as df = df.rename(...)

df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1, inplace = True)
