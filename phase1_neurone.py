import numpy as np

X = np.array([
    [0.2, 0.1],
    [0.8, 0.9],
    [0.3, 0.7],
    [0.9, 0.2],
])
y = np.array([0, 1, 1, 0])

def sigmoid(x):
    # Écrase n'importe quel nombre réel dans (0, 1) : une probabilité
    return 1 / (1 + np.exp(-x))

def forward(X, w, b):
    # Produit scalaire entrées/poids + biais pour tout le batch d'un coup (pas de boucle), puis sigmoid -> probabilité par exemple
    z = np.dot(X, w) + b
    return sigmoid(z)

def compute_loss(y_true, y_pred):
    # Empêche log(0) = -infini si le modèle prédit pile 0 ou 1 (cas limite)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    # Binary Cross-Entropy : "surprise" moyenne face à la vraie étiquette (une erreur confiante coûte cher)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Poids fixés
w = np.array([0.5, -0.3])
b = 0.1

# Forward : calcule la prédiction du neurone pour les 4 points avec les poids fixés
y_pred = forward(X, w, b)

# Compare la prédiction à la vraie étiquette et mesure l'erreur globale du neurone
loss = compute_loss(y, y_pred)

# Affiche les prédictions arrondies (comparaison visuelle avec y) et la loss finale
print("Prédictions :", y_pred.round(3))
print("Étiquettes  :", y)
print("Loss BCE    :", round(loss, 4))
