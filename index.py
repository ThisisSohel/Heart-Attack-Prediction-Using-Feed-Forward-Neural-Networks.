import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("health_data.csv")
data_find = LabelEncoder()
data = data.fillna(0)
data.gender = data_find.fit_transform(data.gender)
data.Married_S = data_find.fit_transform(data.Married_S)
data.Work_T = data_find.fit_transform(data.Work_T)
data.Residence_T = data_find.fit_transform(data.Residence_T)
data.smoking_S = data_find.fit_transform(data.smoking_S)
train, test = train_test_split(data, test_size=0.4)
main_fun = MLPClassifier(activation='logistic', max_iter=1000, hidden_layer_sizes=(30, 30, 30, 30, 30), learning_rate_init = 0.01, solver='sgd')
x_train = train[data.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]
y_train = train[data.columns[11]]

x_test = test[data.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]
y_test = test[data.columns[11]]
main_fun = main_fun.fit(x_train, y_train)
y_pred = main_fun.predict(x_test)
Accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy of Dataset: ", round(Accuracy, 2), "%")