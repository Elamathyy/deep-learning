#code:
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# XOR input and output (Test Case Set 2)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Initialize and train the Perceptron
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)

# Predictions and remarks based on Test Case Set 2
print("Predictions:", clf.predict(X))

# Evaluation based on the Test Case Set 2 Expected Results
for i in range(len(X)):
    prediction = clf.predict([X[i]])[0]
    actual = y[i]
    remark = "Correct" if prediction == actual else "May fail"
    print(f"Input {X[i]} => Predicted Output: {prediction} (Expected: {actual}) - {remark}")

# Plotting the data points and decision boundary
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='red')  # Actual output 0
    else:
        plt.scatter(X[i][0], X[i][1], color='blue')  # Actual output 1

# Plot the decision boundary
x_values = np.array([0, 1])
y_values = -(clf.coef_[0][0] * x_values + clf.intercept_) / clf.coef_[0][1]  # Perceptron decision boundary
plt.plot(x_values, y_values, label="Decision Boundary", color='green')

# Plot settings
plt.title('Perceptron Decision Boundary for XOR')
plt.xlabel('X1')<img width="1598" height="619" alt="Screenshot 2025-08-13 112019" src="https://github.com/user-attachments/assets/31064816-455e-40dc-9a8b-aaea085599a5" />
<img width="1599" height="659" alt="Screenshot 2025-08-13 112008" src="https://github.com/user-attachments/assets/f0481222-0ed2-4620-b038-b550b9d9c885" />
<img width="1593" height="582" alt="Screenshot 2025-08-13 111958" src="https://github.com/user-attachments/assets/bac6cfd7-5800-4ba7-be64-9571d93d7250" />

plt.ylabel('X2')
plt.grid(True)
plt.legend()
plt.show()


#code with output(screenshot):
