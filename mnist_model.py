import numpy as np
import keras
from keras import layers

# Data parameters
def load_mnist_data():
    num_classes = 10
    input_shape = (28, 28, 1)

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

def create_model(x_train, y_train, batch_size = 128, epochs = 15, input_shape = (28, 28, 1), num_classes = 10):
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
    )
    # Train
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    return model

def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    return score

def train_model():
    num_classes = 10
    input_shape = (28, 28, 1)
    batch_size = 128
    epochs = 15
    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    # Create model
    model = create_model(x_train, y_train, batch_size, epochs, input_shape, num_classes)
    # Evaluate
    score = evaluate_model(model, x_test, y_test)

    return model, score

def load_pretrained_model():
    return keras.models.load_model("mnist_model.keras")

if __name__ == "__main__":
    model, score = train_model()
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    model.save("mnist_model.keras")
    print("Model saved to mnist_model.keras")