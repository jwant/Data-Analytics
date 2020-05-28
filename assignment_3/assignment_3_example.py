import sklearn.preprocessing as prep
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


df_test = pd.read_csv(r'test_pub.csv')
df = pd.read_csv(r'train.csv')

df_onehot = pd.get_dummies(df)

keys = df_onehot.keys()
data_keys = [k for k in keys
    if '?' not in k and k[-3:] != "50K"]
data_train = df_onehot[data_keys]
target_train = df_onehot["Salary_ >50K"]

df_onehot1 = pd.get_dummies(df_test)
# add all zero to non-existing keys
for k in data_keys:
    if k not in df_onehot1.keys():
        df_onehot1[k] = 0

data_test = df_onehot1[data_keys]

sc = prep.MinMaxScaler()
data_train_s = sc.fit_transform(data_train)
data_test_s = sc.transform(data_test)

lr = LogisticRegression()
lr.fit(data_train_s, target_train)
# Predict the probability of positive class
pred_test_prob = lr.predict_proba(data_test_s)[:, 1] #

df_test["Predicted"] = pred_test_prob
df_test[["ID","Predicted"]].to_csv("LogisticReg_v0.csv", index=False)

