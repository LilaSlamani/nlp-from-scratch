import numpy as np

X = np.array([[0.2, 0.1], [0.8, 0.9], [0.3, 0.7], [0.9, 0.2]])
y = np.array([0, 1, 1, 0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train(learning_rate, n_epochs=51):
    np.random.seed(42)
    w = np.random.randn(2) * 0.01
    b = 0.0
    for epoch in range(n_epochs):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)
        loss = compute_loss(y, y_pred)
        error = y_pred - y
        dw = np.dot(X.T, error) / len(y)
        db = np.mean(error)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if epoch % 10 == 0:
            print(f"  epoch {epoch:2d} : loss = {loss:.4f}")

# Cas limite : learning_rate = 0 -> rien ne doit bouger, la loss reste identique partout
print("--- Cas limite : learning_rate = 0.0 ---")
train(learning_rate=0.0)

# Scenario adversarial : learning_rate = 10.0 -> pas de correction beaucoup trop grand, la loss doit osciller
print("\n--- Scenario adversarial : learning_rate = 10.0 ---")
train(learning_rate=10.0)
