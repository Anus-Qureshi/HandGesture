#from tensorflow.keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


img_size = 64
batch = 8   # small dataset hai is liye 8

# Training data generator
train_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=(img_size, img_size),
    batch_size=batch,
    class_mode="categorical"
)

# Validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)

val_data = val_datagen.flow_from_directory(
    "dataset/val",
    target_size=(img_size, img_size),
    batch_size=batch,
    class_mode="categorical"
)

# CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(64,64,3)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

model.save("model.h5")

print("✅ Model trained and saved as model.h5")
