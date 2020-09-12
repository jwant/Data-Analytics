import sklearn.preprocessing as prep
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from sklearn.metrics import roc_curve, roc_auc_score


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

data_train.to_csv("preprocessed_train.csv", index=False)

def return_feature_importance(keys,clf):
    key_importance = []
    for i in range(len(keys)):
        key_importance += [(keys[i], clf.feature_importances_[i])]
    return sorted(key_importance, key=lambda a: a[1])


# Intialise Data and function
from sklearn.ensemble import RandomForestClassifier as rdmfrst

feature_list = ['Age','Education years','Capital gain','Capital loss','Race_ White','Native country_ United-States']

forest_data = data_train[feature_list]
n_training_samples = int(len(forest_data) *.75)
n_validation_samples = len(forest_data) - n_training_samples

forest_train = forest_data.head(n_training_samples)
forest_train_target = target_train.head(n_training_samples)
forest_validation = forest_data.tail(n_validation_samples)
forest_validation_target = target_train.tail(n_validation_samples)
forest_target = target_train


# In[200]:


# Build Classifier
clf = rdmfrst(n_estimators=100,max_depth=2, random_state=0)


# In[211]:


# Build Model
clf.fit(X=forest_train,y=forest_train_target)
forest_train_probabilities = clf.predict_proba(forest_train)[:,1]
roc_auc_score(y_true=forest_train_target, y_score=forest_train_probabilities)
return_feature_importance(forest_train.keys(), clf)


# In[210]:
    


# In[194]:


# Validate Model
clf.fit(X=forest_validation,y=forest_validation_target)
forest_validation_probabilities = clf.predict_proba(forest_validation)[:,1]
roc_auc_score(y_true=forest_validation_target, y_score=forest_validation_probabilities)


# In[202]:


# Output
forest_test_probabilities = clf.predict_proba(data_test[feature_list])[:,1]
df_test['Predicted'] = forest_test_probabilities
data_test[["ID","Predicted"]].to_csv("random_forest_v0.csv", index=False)


# In[ ]:




