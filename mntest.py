# use only for testing CNN 
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        pass 
else:
    pass

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the images
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Define the CNN model
def cnn(input_shape, num_classes):
    # model = Sequential()
    # model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding = 'same'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3),  strides=(1, 1), activation='relu', padding = 'same'))
    
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(num_classes, activation='softmax'))
    # return model
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, kernel_size=(3,3),  activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Create and compile the model
input_shape = (28, 28, 1)
num_classes = 10
model = cnn(input_shape, num_classes)
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Training parameters
batch_size = 64
epochs = 10
steps_per_epoch = 12000 // batch_size
validation_steps = test_images.shape[0] // batch_size
subset_train_images = train_images[:12000]
subset_train_labels = train_labels[:12000]
# Training loop
for epoch in range(epochs):
    # Train for one epoch
    for step in range(steps_per_epoch):
        batch_start = step * batch_size
        batch_end = batch_start + batch_size
        x_batch_train = train_images[batch_start:batch_end]
        y_batch_train = train_labels[batch_start:batch_end]
        model.train_on_batch(x_batch_train, y_batch_train)

    # Evaluate on the training set
    train_loss, train_acc = model.evaluate(subset_train_images, subset_train_labels, verbose=0)

    # Evaluate on the validation set
    val_loss, val_acc = model.evaluate(test_images, test_labels, verbose=0)

    # Print metrics
    print(f'Epoch {epoch + 1}/{epochs}')
    print(f' - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
