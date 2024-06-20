import joblib
import os
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score


current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../../data')

X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))

# converte o y_train para o array de 1D
y_train = y_train.values.ravel()

dtc = tree.DecisionTreeClassifier()
dtc.fit(X_train, y_train)
pred = dtc.predict(X_test)

print((accuracy_score(pred, y_test))*100)

joblib.dump(dtc, "models/rf/iris_rf.joblib")
