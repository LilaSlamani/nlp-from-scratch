# Deep Learning J1 : PMC from scratch

Réseau de neurones codé à la main en numpy (zéro Keras, zéro `model.fit()`), puis comparé à Keras.

## Phase 1 : neurone unique, forward pass et calcul d'erreur

Fichier : `phase1_neurone.py`

Neurone unique avec poids fixés (`w = [0.5, -0.3]`, `b = 0.1`, pas d'entraînement dans cette phase). Objectif : implémenter `sigmoid`, `forward` et `compute_loss` (Binary Cross-Entropy), et vérifier leur comportement sur trois scénarios.

### Scénario normal

Poids fixes du cours sur les 4 points d'entraînement.

```
Prédictions : [0.542 0.557 0.51  0.62 ]
Étiquettes  : [0 1 1 0]
Loss BCE    : 0.7519
```

### Cas limite : `X = np.zeros((4, 2))`

Quand toutes les entrées sont nulles, `z = X @ w + b = b` pour chaque exemple : le biais seul pilote la sortie, peu importe `w`.

```
sigmoid(b) = sigmoid(0.1) = [0.52497919 0.52497919 0.52497919 0.52497919]
```

Résultat cohérent : les 4 prédictions sont identiques et égales à `sigmoid(0.1)`. Le code ne plante pas.

### Scénario adversarial : `w = np.zeros(2)`, `b = 0`

Sans poids ni biais, `z = 0` pour tous les exemples → toutes les prédictions convergent vers 0.5 (indécision totale).

```
Prédictions : [0.5 0.5 0.5 0.5]
Loss BCE    : 0.6931471805599453   (≈ -log(0.5))
```

C'est le pire point de départ possible : le réseau ne discrimine rien. C'est aussi pourquoi, en phase 2, les poids sont initialisés avec de petites valeurs aléatoires (`np.random.randn(2) * 0.01`) plutôt qu'à zéro : pour "briser la symétrie" et permettre l'apprentissage.
