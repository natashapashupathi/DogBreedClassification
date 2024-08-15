import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv('labels.csv')

# Add the full path to the image files
df['filename'] = df['id'].apply(lambda x: os.path.join('train', f'{x}.jpg'))

# Splitting the data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# # ImageDataGenerator with augmentation for training
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# # Only rescaling for validation
# val_datagen = ImageDataGenerator(rescale=1./255)

# # Create the generators
# train_generator = train_datagen.flow_from_dataframe(
#     train_df,
#     x_col='filename',
#     y_col='breed',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical'
# )

# val_generator = val_datagen.flow_from_dataframe(
#     val_df,
#     x_col='filename',
#     y_col='breed',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical'
# )

# # Model definition (example, modify according to your model)
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=10
# )

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16  # Use a pretrained VGG16 model
from tensorflow.keras import layers, models, optimizers, callbacks

# Image Data Generator with Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split for validation
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col='breed',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col='breed',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Load Pretrained VGG16 Model (excluding top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the base model
base_model.trainable = False

# Adding custom layers on top of VGG16
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  # Add dropout to prevent overfitting
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
optimizer = optimizers.Adam(learning_rate=1e-4)  # Use a smaller learning rate
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Learning Rate Scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_callback = callbacks.LearningRateScheduler(lr_scheduler)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,  # Increase the number of epochs
    validation_data=validation_generator,
    callbacks=[lr_callback]
)

# Optionally, unfreeze some layers and fine-tune
base_model.trainable = True
for layer in base_model.layers[:15]:
    layer.trainable = False

# Recompile and continue training
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history_finetune = model.fit(
    train_generator,
    epochs=10,  # Fine-tune with more epochs
    validation_data=validation_generator,
    callbacks=[lr_callback]
)


# Save the model
model.save('dog_breed_model.h5')

# Save class labels
import numpy as np
np.save('class_labels.npy', list(train_df.class_indices.keys()))

