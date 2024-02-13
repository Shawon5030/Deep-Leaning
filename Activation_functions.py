import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu(x, alpha=1.6732632423543772848170429916717,
         scale=1.0507009873554804934193349852946):
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def softmax(x):
    exp_vals = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

def swish(x):
    return x * sigmoid(x)

def linear(x):
    return x

def parametric_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def gelu(x):
    return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# Create x values
x = np.linspace(-3, 3, 100)

# Calculate y values for each activation function
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_elu = elu(x)
y_selu = selu(x)
y_softmax = softmax(x)
y_swish = swish(x)
y_linear = linear(x)
y_parametric_relu = parametric_relu(x)
y_gelu = gelu(x)

# Create subplots for each activation function
fig, axs = plt.subplots(4, 3, figsize=(15, 12))
axs[0, 0].plot(x, y_sigmoid)
axs[0, 0].set_title('Sigmoid')
axs[0, 0].grid(True)
axs[0, 1].plot(x, y_tanh)
axs[0, 1].set_title('Tanh')
axs[0, 1].grid(True)
axs[0, 2].plot(x, y_relu)
axs[0, 2].set_title('ReLU')
axs[0, 2].grid(True)
axs[1, 0].plot(x, y_leaky_relu)
axs[1, 0].set_title('Leaky ReLU')
axs[1, 0].grid(True)
axs[1, 1].plot(x, y_elu)
axs[1, 1].set_title('ELU')
axs[1, 1].grid(True)
axs[1, 2].plot(x, y_selu)
axs[1, 2].set_title('SELU')
axs[1, 2].grid(True)
axs[2, 0].plot(x, y_softmax)
axs[2, 0].set_title('Softmax')
axs[2, 0].grid(True)
axs[2, 1].plot(x, y_swish)
axs[2, 1].set_title('Swish')
axs[2, 1].grid(True)
axs[2, 2].plot(x, y_linear)
axs[2, 2].set_title('Linear')
axs[2, 2].grid(True)
axs[3, 0].plot(x, y_parametric_relu)
axs[3, 0].set_title('Parametric ReLU')
axs[3, 0].grid(True)
axs[3, 1].plot(x, y_gelu)
axs[3, 1].set_title('GELU')
axs[3, 1].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Generate a small dataset
np.random.seed(0)
X = np.linspace(-5, 5, 100)
y_true = 3 * X + 2 + np.random.normal(0, 2, 100)  # Linear relationship with noise

# Visualize the dataset with each activation function
plt.figure(figsize=(15, 10))

plt.scatter(X, y_true, label='True Relationship', color='black', alpha=0.5)
plt.plot(X, y_sigmoid, label='Sigmoid', linestyle='--')
plt.plot(X, y_tanh, label='Tanh', linestyle='--')
plt.plot(X, y_relu, label='ReLU', linestyle='--')
plt.plot(X, y_leaky_relu, label='Leaky ReLU', linestyle='--')
plt.plot(X, y_elu, label='ELU', linestyle='--')
plt.plot(X, y_selu, label='SELU', linestyle='--')
plt.plot(X, y_softmax, label='Softmax', linestyle='--')
plt.plot(X, y_swish, label='Swish', linestyle='--')
plt.plot(X, y_linear, label='Linear', linestyle='--')
plt.plot(X, y_parametric_relu, label='Parametric ReLU', linestyle='--')
plt.plot(X, y_gelu, label='GELU', linestyle='--')

plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Visualization of Dataset with Different Activation Functions')
plt.legend()
plt.grid(True)
plt.show()