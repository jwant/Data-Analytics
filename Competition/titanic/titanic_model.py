import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

features = ["Pclass","Sex","SibSp","Parch"]
x_train = pd.get_dummies(train_data[features])
x_test = pd.get_dummies(test_data[features])


model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(x_train,train_data['Survived'])
predictions = model.predict(x_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('predictions.csv', index=False)