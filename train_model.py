import tensorflow as tf
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.models import Sequential
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import os



EMOTIONS = ['angry', 'fear', 'happy', 'sad', 'surprise']
IMG_SIZE = 48
BATCH = 32
EPOCHS = 25

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True
)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    classes=EMOTIONS,
    batch_size=BATCH,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    'dataset/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    classes=EMOTIONS,
    batch_size=BATCH,
    class_mode='categorical'
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

os.makedirs('model', exist_ok=True)
model.save('model/emotion_cnn.keras')
print("✅ Model saved to model/emotion_cnn.keras")