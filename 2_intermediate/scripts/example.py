import torch
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Example using PyTorch
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(4, 10)
        self.fc2 = torch.nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)

# Initialize and train the model
model = SimpleNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

model_tf = keras.Sequential(
    [keras.layers.Dense(10, activation="relu", input_shape=(4,)), keras.layers.Dense(3)],
)

model_tf.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model_tf.fit(X_train, y_train, epochs=100)

# Example using scikit-learn
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

# Evaluate the models
torch_model_eval = model(X_test)
_, predicted = torch.max(torch_model_eval.data, 1)

accuracy = np.mean(predicted.numpy() == y_test)
print(f"PyTorch model accuracy: {accuracy * 100:.2f}%")

tf_loss, tf_accuracy = model_tf.evaluate(X_test, y_test)
print(f"TensorFlow model accuracy: {tf_accuracy * 100:.2f}%")

rf_accuracy = rf_model.score(X_test, y_test)
print(f"Scikit-learn model accuracy: {rf_accuracy * 100:.2f}%")
