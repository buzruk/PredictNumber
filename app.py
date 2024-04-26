from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np
import os

# Load MNIST dataset (replace with your data if not using MNIST)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data (assuming MNIST format)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # Reshape for CNN (add channel dimension)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32') / 255  # Normalize pixel values
x_test = x_test.astype('float32') / 255

# One-hot encode labels (optional, but recommended for categorical classification)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the neural network model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))  # 10 output units for 10 digits

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model on test data
loss, accuracy = model.evaluate(x_test, y_test)
print('Test accuracy:', accuracy)

# Use the trained model for prediction on new images
def predict_digit(image):
  image = np.array(image)
  # Preprocess the new image (same steps as before)
  image = image.reshape(1, 28, 28, 1)
  image = image.astype('float32') / 255
  # Make prediction
  prediction = model.predict(image)
  # Return the most likely digit (index of max value)
  return np.argmax(prediction)

# Initialize a list to store digit counts
digit_counts = [0] * 10  # List of 10 elements initialized to 0
image_folder = "digits"

# Loop through all files in the image folder
for filename in os.listdir(image_folder):
  # Check if the file is a JPG image
  if filename.endswith(".jpg"):
    filepath = os.path.join(image_folder, filename)
    try:
      # Open the image
      image = Image.open(filepath)

      # Predict the digit using your trained model
      new_width = 28
      new_height = 28
      resized_image = image.resize((new_width, new_height))
      predicted_digit = predict_digit(resized_image)  # Assuming predict_digit takes a preprocessed image

      # Increment the count for the predicted digit
      digit_counts[predicted_digit] += 1

    except (IOError, ValueError):  # Handle potential errors like invalid files or filenames
      print(f"Error processing file: {filename}")

print("Digit counts:", digit_counts)
