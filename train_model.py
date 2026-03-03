from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# load dataset
iris = load_iris()
X = iris.data
y = iris.target

# train model
model = RandomForestClassifier()
model.fit(X, y)

# save model
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Train xong va da luu file iris_model.pkl")