import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras


def generate_signals(n_samples=5000, length=200, noise=0.1):
    """Generate synthetic 1D signals made of two sine waves plus noise."""
    t = np.linspace(0, 1, length)
    signals = []
    for _ in range(n_samples):
        freq1 = np.random.uniform(1, 5)
        freq2 = np.random.uniform(6, 12)
        amp1 = np.random.uniform(0.5, 1.0)
        amp2 = np.random.uniform(0.5, 1.0)
        phase1 = np.random.uniform(0, 2 * np.pi)
        phase2 = np.random.uniform(0, 2 * np.pi)
        signal = (
            amp1 * np.sin(2 * np.pi * freq1 * t + phase1)
            + amp2 * np.sin(2 * np.pi * freq2 * t + phase2)
        )
        signal += noise * np.random.randn(length)
        signals.append(signal.astype(np.float32))
    return np.array(signals)


def build_autoencoder(input_dim):
    """Create a slightly larger autoencoder model."""
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim,)),
        keras.layers.Dense(128, activation="selu"),
        keras.layers.Dense(64, activation="selu"),
        keras.layers.Dense(32, activation="selu"),
        keras.layers.Dense(16, activation="selu", name="latent"),
        keras.layers.Dense(32, activation="selu"),
        keras.layers.Dense(64, activation="selu"),
        keras.layers.Dense(128, activation="selu"),
        keras.layers.Dense(input_dim),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    # Generate more complex synthetic data
    signals = generate_signals()
    X_train, X_valid = train_test_split(signals, test_size=0.2, random_state=42)

    model = build_autoencoder(signals.shape[1])

    history = model.fit(
        X_train,
        X_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_valid, X_valid),
        verbose=2,
    )

    # Plot loss curves
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()

    # Compute reconstruction error on the validation set
    reconstructions = model.predict(X_valid)
    mse = np.mean(np.square(reconstructions - X_valid))
    print(f"Reconstruction MSE on validation set: {mse:.6f}")


if __name__ == "__main__":
    main()
