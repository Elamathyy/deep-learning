#code:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# XOR input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Initialize and train the Perceptron model
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)

print("Predictions:", clf.predict(X))
      
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='red', label='Class 0' if i == 0 else "")
    else:
        plt.scatter(X[i][0], X[i][1], color='blue', label='Class 1' if i == 1 else "")

x_values = np.array([0, 1])
y_values = -(clf.coef_[0][0] * x_values + clf.intercept_) / clf.coef_[0][1]
plt.plot(x_values, y_values, color='green', linestyle='--', label='Decision Boundary')

plt.title('Perceptron Decision Boundary for XOR')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True)
plt.show()

#output with code(screenshot):<img width="1599" height="625" alt="Screenshot 2025-08-13 105417" src="https://github.com/user-attachments/assets/dc85bd42-91a1-4148-b3be-e07dc88997ea" />
<img width="1594" height="643" alt="Screenshot 2025-08-13 105429" src="https://github.com/user-attachments/assets/d64919b7-37e2-4d73-923d-9a5c866efbe3" />
