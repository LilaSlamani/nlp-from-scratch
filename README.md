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

## Phase 3 : XOR, échec du neurone unique, victoire de la couche cachée

Fichiers : `phase3_xor.py` (entraînement) et `phase3_qualite_tests.py` (architecture 2-1-1 et bruit sur les coordonnées).

XOR n'est pas linéairement séparable (les classes sont sur les diagonales opposées du carré) : un neurone unique ne peut donc jamais le résoudre, quelle que soit la façon dont on l'entraîne. Une couche cachée (architecture 2-2-1) le résout : chaque neurone caché apprend sa propre séparation linéaire, et la couche de sortie les combine en une frontière non linéaire.

### Scénario normal

Réseau 2-2-1, `learning_rate = 0.5`, 10 000 epochs.

```
Epoch     0 | Loss: 0.7516 | Accuracy: 50.00%
Epoch  2000 | Loss: 0.0466 | Accuracy: 100.00%
Epoch  4000 | Loss: 0.0081 | Accuracy: 100.00%
Epoch  6000 | Loss: 0.0043 | Accuracy: 100.00%
Epoch  8000 | Loss: 0.0030 | Accuracy: 100.00%
Loss finale : 0.0022
Accuracy finale : 100.00%
```

Accuracy 100% bien avant 10 000 epochs, frontière de décision non linéaire visible sur `phase3_xor_boundary.png`.

### Cas limite : architecture 2-1-1 (1 seul neurone caché)

Un seul neurone caché ne peut tracer qu'une seule séparation linéaire, et XOR a besoin d'au moins 2 séparations combinées pour être reconstruit. Avec 1 seul neurone caché, le réseau retombe sur la même limite qu'un neurone unique (Phase 1-2).

```
y_pred = [0.5006 0.4963 0.5038 0.4994]  vs  y_true = [0 1 1 0]
loss finale = 0.6931  |  accuracy finale = 50.00%
```

Le réseau ne discrimine rien (prédictions collées à 0.5, loss égale à `-log(0.5)`, accuracy au niveau du hasard). Voir `phase3_qualite_boundary_2-1-1.png` : la frontière est quasi uniforme, sans découpage visible.

### Scénario adversarial : 5% de bruit sur les coordonnées XOR

```
y_pred = [0.0012 0.4996 0.9983 0.5008]  vs  y_true = [0 1 1 0]
loss finale = 0.3479  |  accuracy finale = 50.00%
```

Le réseau reste très confiant et correct sur 2 points (`0.0012` et `0.9983`), mais devient totalement indécis sur les 2 autres (`0.4996` et `0.5008`, quasiment pile sur la frontière). Le bruit a déplacé exactement les 2 points de la classe "1" dans la zone d'incertitude apprise par le réseau, ce qui donne une loss moyenne modérée (tirée vers le bas par les 2 points sûrs) mais une accuracy à 50% (2 corrects sur 4). Voir `phase3_qualite_boundary_bruit.png` pour la frontière déformée par rapport au scénario normal.

## Phase 4 : spirale 2D, frontière non linéaire visualisée

Fichiers : `phase4_spirale.py` (entraînement) et `phase4_qualite_tests.py` (architecture réduite et bruit fort).

Dataset en spirale (deux classes entrelacées, 400 points), réseau 2-64-64-1 (2 couches cachées ReLU, sortie sigmoid), initialisation He.

**Note** : le cours indique `lr=0.01`, mais avec un gradient divisé par `n` (comme la formule l'indique), la convergence est bien trop lente et devient même instable après 2000 epochs. Comme pour la Phase 2, le corrigé du cours semble utiliser un gradient non divisé par `n`. `lr=0.5` reproduit fidèlement la trajectoire attendue par le cours.

### Scénario normal

Réseau 2-64-64-1, `lr=0.5`, 2000 epochs.

```
Epoch    0 | Loss: 0.7175 | Accuracy: 48.50%
Epoch  500 | Loss: 0.5730 | Accuracy: 66.25%
Epoch 1000 | Loss: 0.3918 | Accuracy: 75.75%
Epoch 1500 | Loss: 0.2668 | Accuracy: 94.25%
Loss finale : 0.0129
Accuracy finale : 100.00%
```

Accuracy largement au-dessus des 90% attendus. La frontière (`phase4_spirale.png`) suit bien la forme des deux spirales.

### Cas limite : architecture 2-2-1 (une seule couche cachée de 2 neurones)

Au lieu des 2 couches de 64 neurones, une seule couche de 2 neurones : beaucoup moins de représentations disponibles pour épouser une forme aussi tordue que la spirale.

```
loss finale = 0.6783  |  accuracy finale = 57.50%
```

Sous-apprentissage net : la frontière (`phase4_qualite_boundary_2-2-1.png`) reste grossière et ne suit pas les spirales, loin des 100% du réseau complet.

### Scénario adversarial : `noise = 0.5` (au lieu de 0.15)

```
loss finale = 0.0544  |  accuracy finale = 98.50%
```

Résultat plus robuste qu'attendu : même avec un bruit fort sur la génération des spirales, le réseau 2-64-64-1 reste à 98.5% d'accuracy, à peine en dessous du scénario propre (100%). Contrairement à l'hypothèse du cours (une dégradation notable en dessous de 90%), ce réseau a manifestement assez de capacité (64 neurones par couche cachée) pour s'adapter même à une version très bruitée de la spirale. Voir `phase4_qualite_boundary_bruit05.png` : la frontière est plus irrégulière que le scénario normal, mais reste globalement fidèle à la forme des spirales.
