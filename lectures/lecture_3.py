import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

db = datasets.load_iris()
X, y = db['data'], db['target']
# U = X[10:20, 1:3]
df = pd.DataFrame(data={'sl':X[:,0], 'sw': X[:,1], 'pl': X[:,2], 'pw':X[:,3]})
fig = px.scatter(df, x='sl', y='pl')
fig.show()