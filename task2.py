# Task 2: MNIST Digit Classification using TensorFlow (CNN Model)

# Step 1: Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Step 3: Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Step 4: Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Step 5: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
print("\n🧠 Training the CNN model...")
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Step 7: Evaluate the model
print("\n📊 Evaluating on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Final Test Accuracy: {test_acc:.4f} (Goal: >95%)")

# Step 8: Visualize predictions
print("\n🔍 Visualizing predictions on 5 test images...")
num_samples = 5
sample_images = x_test[:num_samples]
sample_labels = y_test[:num_samples]
predictions = model.predict(sample_images)

plt.figure(figsize=(12, 4))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap="gray")
    plt.title(f"True: {sample_labels[i]}\nPred: {np.argmax(predictions[i])}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# Step 9: Print prediction results
print("\n🧾 Prediction Summary:")
for i in range(num_samples):
    print(f"Image {i+1} - True Label: {sample_labels[i]}, Predicted: {np.argmax(predictions[i])}")
