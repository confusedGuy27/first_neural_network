import numpy as np 
import nnfs
from nnfs.datasets import spiral_data
import random
nnfs.init()

seed_number = random.randint(0,10000)
np.random.seed(seed_number)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self,dvaluse):
        self.dweights = np.dot(self.inputs.T, dvaluse)
        self.dbiases = np.sum(dvaluse, axis=0, keepdims=True)

        self.dinputs = np.dot(dvaluse,self.weights.T)
class Activation:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    def backward(self,dvalues):
        self.dinputs  = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
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
class SGD:
    def __init__(self, learning_rate=0.0,decay=0.0,weight_decay=0.0,momentum= 0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0 
        self.weight_decay =  weight_decay
        self.momentum = momentum
    def pre_update_parm(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
    def update_params(self,layer):
        dweights = layer.dweights + self.weight_decay * layer.weights
        
        if self.momentum:
            if not hasattr(layer, 'weights_momentums'):
                layer.weights_momentums = np.zeros_like(layer.weights)
                layer.biases_momentums = np.zeros_like(layer.biases)

            biases_updates = self.momentum * layer.biases_momentums - self.current_learning_rate * layer.dbiases
            weight_updates = self.momentum * layer.weights_momentums - self.current_learning_rate * dweights

            layer.weights += weight_updates
            layer.biases += biases_updates

        
        
        
       # layer.weights += -slf.current_learning_rate * dweights
        #layer.biases += -self.current_learning_rate * layer.dbiases
    def after_update(self):
        self.iterations += 1

class dropout:
    def __init__(self,rate):
        self.rate = 1 - rate
    def forward(self,inputs):
        self.mask = np.random.binomial(1,self.rate, size=inputs.shape) /self.rate
        self.output = inputs * self.mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.mask
            
x,y = spiral_data(samples=700, classes = 3)

dense1 = Layer_Dense(2,32)
activation1 = Activation()
dense2 = Layer_Dense(32,32)
activation2 = Activation()
dense3 = Layer_Dense(32,3)
actviaion3 = softmax()
optimizer = SGD(learning_rate = 0.2,decay=1e-4,weight_decay=3e-5, momentum=0.9)
loss_fun = loss_log()

best_acc = 0
best_epoch = 0
for epoch in range(40000):
    
    dense1.forward(x)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    dense3.forward(activation2.output)
    actviaion3.forward(dense3.output)
    loss = loss_fun.calculate(actviaion3.output, y)
    pred = np.argmax(actviaion3.output, axis=1)

    accuracy = np.mean(pred == y)

    samples = len(actviaion3.output)
    dvalues = actviaion3.output.copy()
    dvalues[range(samples), y] -= 1
    dvalues /= samples
    dense3.backward(dvalues)
    
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    optimizer.pre_update_parm()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.after_update()
    if epoch % 100 == 0:
        print(f"epoch: {epoch}, loss: {loss:.3f}, accuracy: {accuracy:.3f}")
    if accuracy >= best_acc:
        best_acc = accuracy
        best_epoch = epoch
print("accuracy",best_acc)
print("epoch",best_epoch)
np.savez(
    "model_weights.npz",
    dense1_weights=dense1.weights, dense1_biases=dense1.biases,
    dense2_weights=dense2.weights, dense2_biases=dense2.biases,
    dense3_weights=dense3.weights, dense3_biases=dense3.biases,
)
print("Saved weights to model_weights.npz")

