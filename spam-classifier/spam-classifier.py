import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("spam.csv", encoding = "ISO-8859-1")

df.rename(columns = {"v1": "label",
                     "v2": "message"}, inplace = True) # this does the same thing as df = df.rename(...)

df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1, inplace = True)
print(df.head())

# See how many spam vs. ham messages there are in the dataset
sns.countplot(df["label"])
plt.title("Spam vs. Ham Messages")
plt.xlabel("Label")
plt.ylabel("Count")

# Encode ham and spam as 0 and 1, respectively
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df["label"])
print(list(le.classes_))
df["label"] = le.transform(df["label"])
print(df.head())

# Train/test split
from sklearn.model_selection import train_test_split
X = df["message"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

