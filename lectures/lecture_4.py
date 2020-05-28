import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

df = pd.read_excel(r'abalone-small-utsonline.xls')
df['height_binned'] = pd.cut(df['Height'], bins=4, labels=['Small','Medium','Large','Huge'])
# df['height_binned'] = pd.qcut(df['Height'], q=4, labels=['Small','Medium','Large','Huge'])
box = px.box(df, y='Height')
box.show()
bins_numeric = pd.qcut(df['Height'], 4, retbins=True)
bar = px.scatter(df, x='height_binned', y='Length', color='Sex')
bar.show()

print(bins_numeric)


