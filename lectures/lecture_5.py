import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


df = pd.read_csv(r'imports-85.csv')
scatter = px.scatter(df,x='horsepower', y='price', color='symboling')
scatter.show()
print(df.describe())

