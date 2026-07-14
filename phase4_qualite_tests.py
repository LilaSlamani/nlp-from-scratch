import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_spiral(n_points=200, noise=0.1, seed=42):
    """Génère deux spirales entrelacées : classe 0 et classe 1."""
    np.random.seed(seed)
    n = n_points // 2
    theta0 = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise
    theta1 = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise + np.pi
    r = np.linspace(0.1, 1.0, n)
    X0 = np.c_[r * np.cos(theta0), r * np.sin(theta0)]
    X1 = np.c_[r * np.cos(theta1), r * np.sin(theta1)]
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X, y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def bce_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train_mlp(X, y, hidden_sizes, n_epochs=2000, lr=0.5, seed=42, boundary_path=None, title=""):
    # Réseau générique : N couches cachées ReLU (tailles données par hidden_sizes) + 1 sortie sigmoid
    np.random.seed(seed)
    sizes = [X.shape[1]] + list(hidden_sizes) + [1]
    Ws, bs = [], []
    for i in range(len(sizes) - 1):
        Ws.append(np.random.randn(sizes[i], sizes[i + 1]) * np.sqrt(2 / sizes[i]))
        bs.append(np.zeros(sizes[i + 1]))

    for epoch in range(n_epochs):
        # Forward : ReLU sur toutes les couches cachées, sigmoid seulement en sortie
        activations = [X]
        zs = []
        for i in range(len(Ws) - 1):
            z = np.dot(activations[-1], Ws[i]) + bs[i]
            zs.append(z)
            activations.append(relu(z))
        z_out = np.dot(activations[-1], Ws[-1]) + bs[-1]
        zs.append(z_out)
        y_pred = sigmoid(z_out).flatten()

        loss = bce_loss(y, y_pred)

        # Backward : même schéma que phase4_spirale.py, généralisé à N couches
        err = (y_pred - y).reshape(-1, 1)
        dWs = [None] * len(Ws)
        dbs = [None] * len(Ws)
        for i in reversed(range(len(Ws))):
            dWs[i] = np.dot(activations[i].T, err) / len(y)
            dbs[i] = np.mean(err, axis=0)
            if i > 0:
                err = np.dot(err, Ws[i].T) * relu_grad(zs[i - 1])

        for i in range(len(Ws)):
            Ws[i] -= lr * dWs[i]
            bs[i] -= lr * dbs[i]

    acc = np.mean((y_pred > 0.5) == y)

    if boundary_path:
        h = 0.02
        xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, h),
                              np.arange(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, h))
        grid = np.c_[xx.ravel(), yy.ravel()]
        a = grid
        for i in range(len(Ws) - 1):
            a = relu(np.dot(a, Ws[i]) + bs[i])
        zg = sigmoid(np.dot(a, Ws[-1]) + bs[-1]).reshape(xx.shape)
        plt.figure(figsize=(7, 6))
        plt.contourf(xx, yy, zg, alpha=0.4, cmap='RdBu')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', s=10, edgecolors='none')
        plt.title(title)
        plt.savefig(boundary_path, dpi=100, bbox_inches='tight')
        plt.close()

    return loss, acc

# Cas limite : sous-apprentissage délibéré, architecture 2-2-1 (une seule couche cachée de 2
# neurones) au lieu de 2-64-64-1. Beaucoup moins de "place" pour représenter une frontière
# tordue : elle doit rester grossière, incapable de suivre le détail des spirales.
X, y = generate_spiral(n_points=400, noise=0.15)
loss_small, acc_small = train_mlp(X, y, hidden_sizes=[2], boundary_path="phase4_qualite_boundary_2-2-1.png", title="Sous-apprentissage (2-2-1)")
print(f"Cas limite (2-2-1)               : loss finale = {loss_small:.4f}  |  accuracy finale = {acc_small:.2%}")

# Scénario adversarial : même architecture 2-64-64-1, mais dataset généré avec beaucoup plus de bruit
X_bruit, y_bruit = generate_spiral(n_points=400, noise=0.5)
loss_bruit, acc_bruit = train_mlp(X_bruit, y_bruit, hidden_sizes=[64, 64], boundary_path="phase4_qualite_boundary_bruit05.png", title="Bruit fort (noise=0.5)")
print(f"Scénario adversarial (noise=0.5) : loss finale = {loss_bruit:.4f}  |  accuracy finale = {acc_bruit:.2%}")
