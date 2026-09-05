import numpy as np 
import nnfs
from nnfs.datasets import spiral_data
import random
nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

        
class Activation:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    
class softmax():
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1 , keepdims=True))
        probabilites = exp_values / np.sum(exp_values, axis=1, keepdims= True)
        self.output = probabilites


class loss:
    def calculate(self,output, y):
        sample_losses = self.forward(output,y) 
        data_loss = np.mean(sample_losses)
        return data_loss
class loss_log(loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            true_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            true_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        log_likelihood = -np.log(true_confidences)
        return log_likelihood


x,y = spiral_data(samples=100, classes = 3)

dense1 = Layer_Dense(2,64)
activation1 = Activation()

dense2 = Layer_Dense(64,64)
activation2 = Activation()

dense3 = Layer_Dense(64,3)
actviaion3 = softmax()
loss_fun = loss_log()

data = np.load("model_weights.npz")

dense1.weights = data["dense1_weights"]
dense1.biases  = data["dense1_biases"]
dense2.weights = data["dense2_weights"]
dense2.biases  = data["dense2_biases"]
dense3.weights = data["dense3_weights"]
dense3.biases  = data["dense3_biases"]

times_to_run = 40
for i in range(times_to_run):
    x,y = spiral_data(samples=100, classes = 3)
    seed_number = random.randint(0,10000)
    np.random.seed(seed_number)

    dense1.forward(x)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    actviaion3.forward(dense3.output)

    loss = loss_fun.calculate(actviaion3.output, y)
    pred = np.argmax(actviaion3.output, axis=1)
    accuracy = np.mean(pred == y)
    print("loss:", loss ,"acc", accuracy)
    
    
