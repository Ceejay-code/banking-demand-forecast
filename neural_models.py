"""Keras neural networks wrapped as scikit-learn classifiers (fit / predict / predict_proba)."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf
import keras
from keras import layers
tf.get_logger().setLevel('ERROR')
from sklearn.base import BaseEstimator, ClassifierMixin

N_MONTHS = 12


class NeuralNetClassifier(ClassifierMixin, BaseEstimator):
    """architecture='mlp': all features -> Dense(2*units) -> Dense(units) -> 3-class softmax
    architecture='gru': a GRU reads the 12 monthly counts in time order; its summary is joined
                        with the other (static) features -> Dense(units) -> 3-class softmax"""

    def __init__(self, architecture='mlp', units=32, dropout=0.2, l2=1e-4, learning_rate=0.001,
                 epochs=100, batch_size=128, patience=10, random_state=42):
        self.architecture = architecture
        self.units = units
        self.dropout = dropout
        self.l2 = l2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def _inputs(self, X):
        X = np.asarray(X, dtype='float32')
        if self.architecture == 'gru':
            return [X[:, :N_MONTHS, None], X[:, N_MONTHS:]]
        return X

    def _build(self, n_features):
        weight_decay = keras.regularizers.l2(self.l2)
        if self.architecture == 'mlp':
            inputs = keras.Input(shape=(n_features,))
            x = layers.Dense(2 * self.units, activation='relu', kernel_regularizer=weight_decay)(inputs)
            x = layers.Dropout(self.dropout)(x)
            x = layers.Dense(self.units, activation='relu', kernel_regularizer=weight_decay)(x)
        else:
            sequence_in = keras.Input(shape=(N_MONTHS, 1))
            static_in = keras.Input(shape=(n_features - N_MONTHS,))
            sequence_summary = layers.GRU(self.units, kernel_regularizer=weight_decay)(sequence_in)
            x = layers.Concatenate()([sequence_summary, static_in])
            x = layers.Dense(self.units, activation='relu', kernel_regularizer=weight_decay)(x)
            inputs = [sequence_in, static_in]
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(3, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X, y):
        keras.utils.set_random_seed(self.random_state)
        self.classes_ = np.array([0, 1, 2])
        self.model_ = self._build(np.asarray(X).shape[1])
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience,
                                                   restore_best_weights=True)
        history = self.model_.fit(self._inputs(X), np.asarray(y), validation_split=0.15,
                                  epochs=self.epochs, batch_size=self.batch_size,
                                  callbacks=[early_stop], verbose=0)
        self.history_ = history.history
        return self

    def predict_proba(self, X):
        return self.model_.predict(self._inputs(X), verbose=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
