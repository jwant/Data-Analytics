import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px



df = pd.read_excel(r'98126764.xlsx')

plot = px.scatter_matrix(df)
plot.show()
