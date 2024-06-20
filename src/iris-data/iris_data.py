import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

iris = load_iris()

X = iris["data"]
y = iris["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

output_dir = os.path.join(os.path.dirname(__file__), "../../data")

os.makedirs(output_dir, exist_ok=True)

pd.DataFrame(X_train).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
pd.DataFrame(X_test).to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
pd.DataFrame(y_train).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
pd.DataFrame(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
