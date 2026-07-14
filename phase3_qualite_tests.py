import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

y_xor = np.array([0, 1, 1, 0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_loss_bce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train_xor(X, hidden_size, n_epochs=10000, learning_rate=0.5, seed=42, boundary_path=None):
    # Même réseau et même backprop que phase3_xor.py, mais avec une taille de couche cachée variable
    np.random.seed(seed)
    W1 = np.random.randn(2, hidden_size) * 0.5
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, 1) * 0.5
    b2 = np.zeros(1)

    for epoch in range(n_epochs):
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)
        y_pred = a2.flatten()

        loss = compute_loss_bce(y_xor, y_pred)

        error2 = y_pred - y_xor
        dW2 = np.dot(a1.T, error2.reshape(-1, 1)) / len(y_xor)
        db2 = np.mean(error2)

        error1 = np.dot(error2.reshape(-1, 1), W2.T) * (a1 * (1 - a1))
        dW1 = np.dot(X.T, error1) / len(y_xor)
        db1 = np.mean(error1, axis=0)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    acc = np.mean((y_pred > 0.5) == y_xor)
    print(f"  y_pred = {y_pred.round(4)}  vs  y_true = {y_xor}")

    if boundary_path:
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        z1g = sigmoid(np.dot(grid, W1) + b1)
        z2g = sigmoid(np.dot(z1g, W2) + b2).reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, z2g, alpha=0.4, cmap='RdBu')
        plt.scatter(X[:, 0], X[:, 1], c=y_xor, s=100, cmap='RdBu', edgecolors='k')
        plt.title(f"XOR : frontière ({hidden_size} neurone(s) caché(s))")
        plt.savefig(boundary_path, dpi=100, bbox_inches='tight')
        plt.close()

    return loss, acc

# Explication: pourquoi l'architecture 2-1-1 ne converge pas sur XOR ?
# Un seul neurone caché ne peut tracer qu'une seule séparation linéaire, et une frontière
# non linéaire comme celle de XOR a besoin d'au moins 2 séparations combinées (une par
# neurone caché) pour être reconstruite. Avec 1 seul neurone caché, le réseau se retrouve
# aussi limité qu'un neurone unique (phase 1-2) : il retombe sur une prédiction de 0.5
# partout, la loss stagne à -log(0.5) ~ 0.693, l'accuracy retombe à 50% (deviner au hasard).
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
loss_211, acc_211 = train_xor(X_xor, hidden_size=1, boundary_path="phase3_qualite_boundary_2-1-1.png")
print(f"Cas limite (2-1-1)      : loss finale = {loss_211:.4f}  |  accuracy finale = {acc_211:.2%}")

# Scénario adversarial : 5% de bruit sur les coordonnées XOR, même architecture 2-2-1 (fonctionnelle).
# Le bruit peut suffire à déplacer un point tout près de la frontière apprise par le réseau.
np.random.seed(0)
X_xor_bruit = X_xor + np.random.randn(*X_xor.shape) * 0.05
loss_bruit, acc_bruit = train_xor(X_xor_bruit, hidden_size=2, boundary_path="phase3_qualite_boundary_bruit.png")
print(f"Scénario adversarial (bruit 5%) : loss finale = {loss_bruit:.4f}  |  accuracy finale = {acc_bruit:.2%}")
