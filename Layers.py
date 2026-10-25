import numpy as np


class Layer:        #class so every layer has the same methods and thingies

    def forward(self,x):            #data flows trough to make a prediction
        raise NotImplementedError

    def backward(self, grad_output):    #error flows through the layers. So it gives how massive the errror is in eacj layer  #Gradouput is for how much the error will change the output of the layer
        raise NotImplementedError
    
    def params(self):                   #returns the parameters of the layer and their gradients
        return[]


class Dense(Layer):                         #The layer with weights and biases. So it is the layer that actually does the learning
    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)   # He initialization for weights
        self.b = np.zeros(n_out)                                      #The biases are initialized to zero
        self.dW = np.zeros_like(self.W)                             #The gradients for the weights are initialized to zero
        self.db = np.zeros_like(self.b)                             #The gradients for the biases are initialized to zero
        self.x = None                                               # The input to the layer is initialized to None

    def forward(self, x):                                           # The forward pass computes the output of the layer given the input x. It stores the input x for use in the backward pass.
        self.x = x                                                  # Store the input for use in the backward pass
        return x @ self.W + self.b                                  # Compute the output of the layer using matrix multiplication and addition of biases

    def backward(self, grad_output):                                #The backward pass computes the gradients of the loss
        self.dW = self.x.T @ grad_output                            # Compute the gradient of the loss with respect to the weights
        self.db = np.sum(grad_output, axis = 0, keepdims=True)      # Compute the gradient of the loss with respect to the biases
        return grad_output @ self.W.T                               # Compute the gradient of the loss with respect to the input x

    def params(self):                                               
        return [(self.W, self.dW), (self.b, self.db)]

class relu(Layer):                                                  #Non linear activation function. Allowing to learn more complex funtions

    def __init__(self):                                             # Initialize the layer
        self.mask = None                                            

    def forward(self, x):                                           # The forward pass computes the output of the layer given the input x.
        self.mask = x > 0                                           
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask

class Sigmoid(Layer):                                               #Numbers between 0 and 1. For outputs like probabilities. It is also a non linear activation function

    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, grad_output):
        return grad_output * self.out * (1 - self.out)


def gradient_check():                                               #Proving that the backprop is working fine
    np.random.seed(0)
    layers = [Dense(3,5), relu(), Dense(5, 2)]                      #Building a small network with 3 input features, 5 hidden units, and 2 output units
    x = np.random.randn(4, 3)

    def forward_all(inp):
        for layer in layers:
            inp = layer.forward(inp)
        return inp

    def loss_fn(out):
        return 0.5 * np.sum(out ** 2)


    out = forward_all(x)                    
    grad = out
    for layer in reversed(layers):                                  
        grad = layer.backward(grad)
    analytic = layers[0].dW.copy()

    eps = 1e-5                                                      
    W = layers[0].W         
    numeric = np.zeros_like(W)
    for i in range(W.shape[0]):                                     # Makes a small perturbation to each weight
        for j in range(W.shape[1]):                                 # For each weight
            original = W[i, j]                                      # Store the original weight
            W[i, j] = original + eps
            loss_plus = loss_fn(forward_all(x))
            W[i, j] = original -eps
            loss_minus = loss_fn(forward_all(x))
            W[i, j] = original
            numeric[i, j] = (loss_plus - loss_minus) / (2 * eps)   # Compute the numerical gradient using central difference formula

    diff = np.max(np.abs(analytic - numeric))                       # Compute the maximum difference between the analytical and numerical gradients
    print(f"Max difference: {diff:.2e}")
    print("OK, Backprop is working" if diff < 1e-6 else "Abort or sth, Backprop left us")   # Print the result of the gradient check, if under 1e-6 it is considered to be working correctly, otherwise it is not.




def try_input():
    # same network as in the gradient check
    layers = [Dense(3, 5), relu(), Dense(5, 2)]

    # Own input: one example with 3 numbers
    values = input("Enter 3 numbers (with space in between): ")
    x = np.array([[float(v) for v in values.split()]])  # Form (1, 3)

    out = x
    for layer in layers:
        out = layer.forward(out)

    print("Input:", x)
    print("Output:", out)




if __name__ == "__main__":                                          # Just for the convenience and some correctness checking
    gradient_check()
    try_input()
    print("All done, Backprop is working fine and the input works too, But the prediction has no value since the network is not trained yet.")

#This took me soooooooo long just to get the info and understanding of the backpropagation process.