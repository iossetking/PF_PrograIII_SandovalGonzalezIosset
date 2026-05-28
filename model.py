import tensorflow as tf

def build_model(n_features):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=[n_features]),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )
    return model

def train_model(X_train, y_train):
    model = build_model(X_train.shape[1])
    print("Training model...")
    history = model.fit(X_train, y_train, epochs=200, validation_split=0.1, verbose=0)
    print("Model trained!")
    return model, history
