import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load the trained model
model = tf.keras.models.load_model('dog_breed_model.h5')

# Load the labels
labels_df = pd.read_csv('labels.csv')

# Add the full path to the image files
labels_df['filename'] = labels_df['id'].apply(lambda x: os.path.join('train', f'{x}.jpg'))

# Filter the dataframe to include only images of classes present in the model
# Assuming the model was trained on 120 classes, include them all in evaluation
# Randomly sample some records to evaluate
subset_size = 10  # Number of records to evaluate
subset_df = labels_df.groupby('breed').apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)

# Load Data for Evaluation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

val_data = datagen.flow_from_dataframe(
    dataframe=subset_df,
    x_col='filename',
    y_col='breed',
    target_size=(150, 150),  # Use the same target size as in training
    batch_size=subset_size,  # Use batch size equal to subset size to evaluate all at once
    class_mode='categorical',
    shuffle=False  # Keep shuffle False to align with true labels
)

# Evaluate the model
loss, accuracy = model.evaluate(val_data)

# Print accuracy
print(f"Accuracy: {accuracy*100:.2f}%")

# Predictions
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(val_data.classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
target_names = list(val_data.class_indices.keys())
report = classification_report(val_data.classes, y_pred_classes, target_names=target_names)
print(report)
