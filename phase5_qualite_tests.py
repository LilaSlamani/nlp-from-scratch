import numpy as np
import tensorflow as tf
from tensorflow import keras
import time

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

def build_model():
    return keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=(784,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])

# Cas limite : epochs=0, aucun entraînement ne doit avoir lieu
print("--- Cas limite : epochs=0 ---")
model = build_model()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
try:
    history = model.fit(X_train, y_train, epochs=0, batch_size=64, validation_split=0.1, verbose=0)
    print("Pas d'erreur levée.")
    print("history.history =", history.history)
    print("Nombre d'epochs enregistrées :", len(history.epoch))
except Exception as e:
    print(f"Erreur levée : {type(e).__name__}: {e}")

# Scénario adversarial : batch_size=1 (SGD pur, mise à jour après chaque exemple)
# 1 seule epoch ici (au lieu de 5) pour garder un temps d'exécution raisonnable
print("\n--- Scénario adversarial : batch_size=1, 1 epoch ---")
tf.random.set_seed(42)
model2 = build_model()
model2.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
start = time.time()
history2 = model2.fit(X_train, y_train, epochs=1, batch_size=1, validation_split=0.1, verbose=1)
elapsed2 = time.time() - start
print(f"\nTemps pour 1 epoch (batch_size=1) : {elapsed2:.1f}s")
print(f"val_accuracy : {history2.history['val_accuracy'][-1]:.4f}")
print(f"val_loss     : {history2.history['val_loss'][-1]:.4f}")
print("\nRappel scénario normal (phase5_keras_mnist.py) : 1 epoch (batch_size=64) tourne en quelques secondes.")
