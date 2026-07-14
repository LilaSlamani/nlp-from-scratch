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

## Phase 2 : descente de gradient à la main, loss par epoch

Fichiers : `phase2_gradient.py` (entraînement) et `phase2_qualite_tests.py` (mêmes fonctions, `learning_rate` variable pour observer son rôle).

Le neurone de la Phase 1 avait des poids fixés à la main. Ici, il les apprend lui-même : à chaque epoch, on calcule le gradient de la loss (BCE + sigmoid, dont la combinaison se simplifie via la chain rule) et on met à jour `w` et `b` par descente de gradient.

### Scénario normal

`learning_rate = 0.1`, 50 epochs.

```
Epoch   0 | Loss: 0.6934
Epoch  10 | Loss: 0.6688
Epoch  20 | Loss: 0.6468
Epoch  30 | Loss: 0.6265
Epoch  40 | Loss: 0.6074
Loss finale : 0.5910
```

La loss décroît de façon monotone (`0.6934 → 0.5910`), la courbe (`phase2_loss_curve.png`) est bien descendante : le critère de réussite de cette phase est rempli.

### Cas limite : `learning_rate = 0.0`

Sans correction, la loss reste figée à sa valeur de départ, epoch après epoch.
```
epoch  0 : loss = 0.6934
epoch 10 : loss = 0.6934
epoch 20 : loss = 0.6934
epoch 30 : loss = 0.6934
epoch 40 : loss = 0.6934
epoch 50 : loss = 0.6934
```
Les poids ne bougent pas non plus. Le code ne plante pas. C'est le plancher de débogage du cours : si une courbe ne descend jamais, la première chose à vérifier est que `learning_rate` n'est pas resté à 0 par erreur.

### Scénario adversarial : `learning_rate = 10.0`

Le pas de correction est bien trop grand, la loss oscille de façon chaotique dans les premières epochs avant de se stabiliser.
```
epoch 0 : loss = 0.6934
epoch 1 : loss = 0.5326
epoch 2 : loss = 0.5970
epoch 3 : loss = 1.0224   (pire qu'au départ)
epoch 4 : loss = 0.7579
epoch 5 : loss = 0.8841
epoch 6 : loss = 0.2941
epoch 7 : loss = 0.1665
epoch 8 : loss = 0.0753
epoch 9 : loss = 0.0695
epoch 50 : loss = 0.0210
```
