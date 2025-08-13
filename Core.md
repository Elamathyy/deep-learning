#code:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
# XOR input and output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
# Model definition
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
history = model.fit(X, Y, epochs=1000, verbose=0)
# Evaluate the model
loss, acc = model.evaluate(X, Y, verbose=0)
print("Accuracy:", acc)
# Make predictions
predictions = model.predict(X)
rounded_predictions = np.round(predictions)
# Display predictions
for i in range(len(X)):
    print(f"Input: {X[i]} -> Prediction: {int(rounded_predictions[i][0])}")
# Plot the loss curve
plt.plot(history.history['loss'])
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

#output with code(screenshot):<img width="1594" height="684" alt="Screenshot 2025-08-13 102119" src="https://github.com/user-attachments/assets/d54ddc73-edbb-426a-a02d-faa7b8bd67f3" />
<img width="1592" height="671" alt="Screenshot 2025-08-13 102057" src="https://github.com/user-attachments/assets/ed378fb8-b106-44da-bfc0-db7ac2abbd8c" />
