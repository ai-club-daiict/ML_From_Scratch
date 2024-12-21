import numpy as np
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iterations):
            y_pred = self.predict(X)
            self.weights = self.weights + self.learning_rate*np.array(np.matmul(X.T, (y - y_pred)))
            self.bias = self.bias + self.learning_rate*np.sum(y - y_pred)
        return self.weights, self.bias
    def loss(self, y_true, y_pred):
        return (np.sum((y_true - y_pred) ** 2))/2
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
X=np.array([[1],[2],[3]])
y=np.array([1,2,3])
lr=LinearRegression()
lr.fit(X,y)
print("Weights: ",lr.weights,"Bias: ",lr.bias)
print("Predictions: ",lr.predict(X))
print("Loss: ",lr.loss(y,lr.predict(X)))