import numpy as np
import tensorflow as tf
from tensorflow import keras
import time

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

def run(hidden_activation, label):
    tf.random.set_seed(42)
    if hidden_activation is None:
        # Cas limite : pas d'argument activation -> Keras applique une activation linéaire (identité)
        layers = [
            keras.layers.Dense(128, input_shape=(784,)),
            keras.layers.Dense(64),
            keras.layers.Dense(10, activation="softmax"),
        ]
    else:
        layers = [
            keras.layers.Dense(128, activation=hidden_activation, input_shape=(784,)),
            keras.layers.Dense(64, activation=hidden_activation),
            keras.layers.Dense(10, activation="softmax"),
        ]
    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    start = time.time()
    history = model.fit(X_train, y_train, epochs=10, batch_size=64,
                         validation_split=0.1, verbose=0)
    train_time = time.time() - start
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    val_loss_final = history.history['val_loss'][-1]
    print(f"{label:10s} | val_loss = {val_loss_final:.4f} | test_accuracy = {test_acc:.4f} | temps = {train_time:.0f}s")

print("=== Cas limite : pas d'activation dans les couches cachées (\"linear\") ===")
run(None, "linear")

print("\n=== Scénario adversarial : softmax dans les couches cachées ===")
# Mauvaise pratique : softmax force ses sorties à sommer à 1, ce qui écrase l'information
# entre les couches. À réserver à la couche de sortie en classification multiclasse.
run("softmax", "softmax")

print("\n=== Rappel relu (phase6_activations.py) ===")
print("relu       | val_loss = 0.1006 | test_accuracy = 0.9778")
