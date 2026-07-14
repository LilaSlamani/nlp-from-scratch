import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X = np.array([[0.2, 0.1], [0.8, 0.9], [0.3, 0.7], [0.9, 0.2]])
y = np.array([0, 1, 1, 0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

np.random.seed(42)
w = np.random.randn(2) * 0.01  # petits poids aléatoires, pour briser la symétrie
b = 0.0
learning_rate = 0.1
n_epochs = 50
losses = []

for epoch in range(n_epochs):
    # Forward pass : prédiction du neurone avec les poids actuels
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)

    # Mesure l'erreur du neurone à cette étape, pour tracer la convergence
    loss = compute_loss(y, y_pred)
    losses.append(loss)

    # Gradient de la loss (BCE+sigmoid se simplifie via la chain rule) : direction et ampleur de la correction pour chaque poids
    error = y_pred - y
    dw = np.dot(X.T, error) / len(y)
    db = np.mean(error)

    # Descente de gradient : avance dans la direction opposée au gradient, d'un pas égal au learning rate
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | w: {w.round(3)} | b: {b:.3f}")

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel("Epoch"); plt.ylabel("Loss BCE")
plt.title("Convergence du neurone unique")
plt.savefig("phase2_loss_curve.png", dpi=100, bbox_inches='tight')
print(f"\nCourbe sauvegardée : phase2_loss_curve.png")
print(f"Loss finale : {losses[-1]:.4f}")
