import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px



df = pd.read_excel(r'98126764.xlsx')
scatter = px.scatter(y=df['Age'], x=df['Education years'])
scatter.show()
print(df['Age'].describe())

# df = pd.DataFrame(data={'sl':X[:,0], 'sw': X[:,1], 'pl': X[:,2], 'pw':X[:,3]})
# fig = px.scatter(df, x='sl', y='pl')
# fig.show()

