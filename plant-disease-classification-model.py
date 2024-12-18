## import all the essential libraries and modules required for computer vision
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model architecture
def define_cnn_model(input_shape, num_classes):
    """
    Defines a rough CNN model architecture.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of classes in the classification problem.

    Returns:
        A TensorFlow Keras model instance.
    """
    model = Sequential()

    # Convolutional block 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional block 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional block 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Define the input shape and number of classes
input_shape = (256, 256)
num_classes = 5

# Build the CNN model
model = define_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=keras.metrics.Accuracy)

# Print the model summary
print(model.summary())
