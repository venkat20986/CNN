Aim:To develop simple LSTM model for sequence classification problem in Python using the Keras deep learning library


ResourcesRequired:

Google CoLab,PCwithWindows1064-bitOS.

Theory:

Sequence classification is a technique that enables machines to understand and categorize different types of data in a sequence. The model is trained on a dataset of labeled sequences and then used to make predictions on new, unseen sequences. In other words: this is a supervised learning problem where the goal is to generalize to new, unseen examples. Common techniques for sequence classification include recurrent neural networks (RNNs), long short-term memory (LSTM) networks, and convolutional neural networks (CNNs).

LSTM networks are well-suited for sequence classification tasks as they have the ability to remember and retain information from previous time steps, which allows them to understand the context of the input sequence and make more accurate predictions. They also have the ability to handle long sequences without the problem of vanishing gradients, which occurs in traditional RNNs, by using a memory cell that can retain information for a longer period of time.











Code
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate random sequences and labels
num_samples = 100
sequence_length = 10
X = np.random.randint(0, 100, size=(num_samples, sequence_length))  # Random sequences of integers

# Model configuration
lstm_units = 64

# Define LSTM model
model = Sequential()
model.add(LSTM(units=lstm_units, input_shape=(sequence_length, 1)))
model.add(Dense(units=1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

y = np.random.randint(0, 2, size=(num_samples,))

# Train model
model.fit(np.expand_dims(X, axis=-1), y, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(np.expand_dims(X, axis=-1), y)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Calculate the sum of the sequence
sequence_sum = np.sum(X)

# Predict the class label using the trained model
predicted_labels = model.predict(np.expand_dims(X, axis=-1))

# Classify each sequence based on the last digit of the sum
predicted_classes = ["even" if sequence_sum % 10 % 2 == 0 else "odd" for sequence_sum in np.sum(X, axis=1)]

# Print the random sequences, their sum, and the predicted class labels
for i in range(len(X)):
sequence_sum = np.sum(X[i])
print("Random Sequence:", X[i])
print("Sequence Sum:", sequence_sum)
print("Predicted Class Label (Based on Sum):", predicted_classes[i])
print()

Procedure
Generate a data consisting of random sequences of integers for binary classification
Build The LSTM model consisting of an Embedding layer, an LSTM layer, and a Dense output layer with a sigmoid activation function
Compile the model using binary cross-entropy loss and the Adam optimizer
Train the generated data for 10 epochs with a batch size of 32
Evaluate the trained model on the same data used for training to classify odd and even number sums
