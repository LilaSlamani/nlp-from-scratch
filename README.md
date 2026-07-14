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

### Scénario normal

Réseau 2-64-64-1, `lr=0.01` (conforme à la consigne), 2000 epochs.

```
Epoch    0 | Loss: 0.7175 | Accuracy: 48.50%
Epoch  500 | Loss: 0.6736 | Accuracy: 51.75%
Epoch 1000 | Loss: 0.6637 | Accuracy: 62.00%
Epoch 1500 | Loss: 0.6556 | Accuracy: 71.25%
Loss finale : 0.6473
Accuracy finale : 71.50%
```

En dessous des 90% attendus par le cours. Avec le gradient divisé par `n` et `lr=0.01`, la convergence est trop lente pour ce problème sur 2000 epochs : la frontière (`phase4_spirale.png`) ne suit que grossièrement les spirales, la loss décroît de façon monotone mais n'a pas fini de converger.

### Cas limite : architecture 2-2-1 (une seule couche cachée de 2 neurones)

Au lieu des 2 couches de 64 neurones, une seule couche de 2 neurones : beaucoup moins de représentations disponibles pour épouser une forme aussi tordue que la spirale.

```
loss finale = 0.6878  |  accuracy finale = 52.50%
```

Sous-apprentissage à peine plus marqué que le scénario normal (déjà lui-même sous-entraîné avec `lr=0.01`) : la frontière (`phase4_qualite_boundary_2-2-1.png`) reste grossière et ne suit pas les spirales.

### Scénario adversarial : `noise = 0.5` (au lieu de 0.15)

```
loss finale = 0.6524  |  accuracy finale = 65.50%
```

Avec `lr=0.01`, le réseau n'a de toute façon pas fini de converger même sur la version propre de la spirale (voir scénario normal), donc la comparaison "90% propre vs bruité" du cours ne s'applique pas directement ici : les deux versions restent sous-entraînées à 2000 epochs. Voir `phase4_qualite_boundary_bruit05.png`.

## Phase 5 : passage à Keras sur MNIST flatten

Fichiers : `phase5_keras_mnist.py` (entraînement) et `phase5_qualite_tests.py` (epochs=0, batch_size=1).

Même logique que les phases précédentes (forward pass, loss, backprop, mise à jour des poids), mais avec Keras : le même problème s'écrit en une quinzaine de lignes au lieu d'une centaine, et `model.fit()` remplace toute la boucle d'entraînement codée à la main. Dataset MNIST (70 000 images de chiffres 28x28, aplaties en vecteurs de 784 valeurs). Architecture `Dense(128, relu) -> Dense(64, relu) -> Dense(10, softmax)`, 109 386 paramètres.

### Scénario normal

`epochs=5`, `batch_size=64`.

```
Epoch 1/5 : val_accuracy = 0.9535
Epoch 2/5 : val_accuracy = 0.9715
Epoch 3/5 : val_accuracy = 0.9743
Epoch 4/5 : val_accuracy = 0.9755
Epoch 5/5 : val_accuracy = 0.9758

Temps d'entraînement : 22.4s
Test accuracy : 0.9713
Test loss : 0.0931
```

`Test accuracy` au-dessus du seuil demandé (`> 0.97`), en 22 secondes contre les centaines/milliers d'epochs qu'aurait demandé l'équivalent en numpy pur.

### Cas limite : `epochs=0`

```
history.history = {}
Nombre d'epochs enregistrées : 0
```

Aucune erreur levée : Keras accepte `epochs=0` et retourne simplement un historique vide, sans entraîner le modèle. C'est l'une des deux issues prévues par le cours (erreur ou historique vide).

### Scénario adversarial : `batch_size=1` (SGD pur)

Une seule epoch testée (au lieu de 5) pour garder un temps d'exécution raisonnable, `batch_size=1` signifiant une mise à jour des poids après chaque exemple individuel (54 000 mises à jour pour 1 epoch).

```
Temps pour 1 epoch (batch_size=1) : 228.8s
val_accuracy : 0.9535
val_loss     : 0.1660
```

Comparé aux ~4-5s par epoch avec `batch_size=64`, `batch_size=1` est environ **50 fois plus lent**, pour un résultat pas meilleur (`val_accuracy` similaire à celle obtenue dès la première epoch en `batch_size=64`). Le coût d'un batch trop petit : chaque exemple déclenche une mise à jour complète des poids (calcul de gradient, appel de l'optimiseur), et ce surcoût par étape domine largement sur des mini-batches d'un seul exemple, sans bénéfice en qualité d'apprentissage.

## Phase 6 : comparaison des fonctions d'activation

Fichier : `phase6_activations.py`. Même architecture (128-64-10), même dataset MNIST, même nombre d'epochs (10), `Adam(lr=1e-3)` conforme à la consigne : seule l'activation des couches cachées change (`sigmoid`, `tanh`, `relu`).

### Résultat

```
Activation | Val loss epoch 10 | Test accuracy | Epoch < 0.1 loss | Temps (s)
sigmoid    | 0.0705             | 0.9760        | 5                 | 50
tanh       | 0.0839             | 0.9747        | 3                 | 50
relu       | 0.1006             | 0.9778        | 2                 | 50
```

`relu` converge le plus vite (sous 0.1 dès l'epoch 2, contre 3 pour `tanh` et 5 pour `sigmoid`), confirmant l'affirmation du cours sur la vitesse initiale. Mais la courbe (`phase6_activations_curve.png`) montre qu'à partir de l'epoch 3-4, `relu` et `tanh` se remettent à sur-apprendre (`val_loss` qui remonte), alors que `sigmoid`, plus lente à démarrer, continue de descendre sans interruption. Résultat à l'epoch 10 : `sigmoid` a la meilleure `val_loss` (0.0705), pas parce qu'elle généralise mieux (`relu` a la meilleure `test_accuracy`, 0.9778), mais parce qu'elle est encore en phase de descente quand `relu`/`tanh` ont déjà dépassé leur point optimal.

### Cas limite : pas d'activation dans les couches cachées (`"linear"`)

Fichier : `phase6_qualite_tests.py`. Sans argument `activation`, Keras applique une activation linéaire (identité) : plusieurs couches linéaires empilées équivalent à une seule couche linéaire.

```
linear | val_loss = 0.2532 | test_accuracy = 0.9175
```

Nettement moins bon que `relu` (`0.1006` / `0.9778`) : sans non-linéarité, le réseau ne peut pas apprendre les patterns complexes des chiffres manuscrits.

### Scénario adversarial : softmax dans les couches cachées

```
softmax | val_loss = 0.3121 | test_accuracy = 0.9189
```

Également dégradé par rapport à `relu`. Softmax force ses sorties à sommer à 1 : utilisée dans une couche cachée, elle écrase artificiellement les valeurs intermédiaires que les couches suivantes ont besoin de recevoir sans contrainte, ce qui nuit à l'apprentissage. Softmax doit rester réservée à la couche de sortie en classification multiclasse.
