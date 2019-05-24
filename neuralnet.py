import numpy as np

class Node:
    
    def __add__(self, other):
        return Add(self, other)
    
    def __mul__(self, other):
        return Mult(self, other)

class Ar(Node):
    
    '''Holds a numpy array
    If passed a shape argument, randomly initializes
    '''
    
    def __init__(self, value = None, shape = None):
        self.value = value
        if shape:
            self.value = np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
        self.grad = None
    
    def get_value(self):
        return self.value
    
    def backprop(self):
        pass

    
class Add(Node):
    
    '''Adds a row (b) to every row of a matrix (A)'''
    
    def __init__(self, A, b):
        self.value = None
        self.grad = None
        self.A = A
        self.b = b
        
    def get_value(self):
        self.value = self.A.get_value() + self.b.get_value()
        return self.value
    
    def backprop(self):
        self.A.grad = np.copy(self.grad)
        self.b.grad = np.sum(self.grad, axis = 0)
        self.A.backprop()
        self.b.backprop()

class Mult(Node):
    
    '''Multiplies a matrix (A) by a matrix (B) to form the product (AB)'''
    
    def __init__(self, A, B):
        self.value = None
        self.grad = None
        self.A = A
        self.B = B
    
    def get_value(self):
        self.value = self.A.get_value().dot(self.B.get_value())
        return self.value
    
    def backprop(self):
        self.A.grad = self.grad.dot(self.B.value.T)
        self.B.grad = self.A.value.T.dot(self.grad)
        self.A.backprop()
        self.B.backprop()

class MSE(Node):
    
    '''mse aka L2 loss
    yhat is the input and is rows of predictions
    y is the ground truth and should have same shape as yhat
    value is the average L2 loss over all rows
    '''
    
    def __init__(self, yhat, y = None):
        self.value = None
        self.grad = 1
        self.yhat = yhat
        self.y = y
    
    def get_value(self):
        n = self.yhat.get_value().shape[0]
        self.value = np.sum(np.square(self.yhat.value - self.y)) / (2 * n)
        return self.value
    
    def backprop(self):
        self.yhat.grad = (self.yhat.value - self.y) / self.yhat.value.shape[0]
        self.yhat.backprop()

class Softmax(Node):
    
    '''row-wise softmax'''
    
    def __init__(self, A):
        self.value = None
        self.grad = None
        self.A = A
    
    def get_value(self):
        A = np.copy(self.A.get_value())
        A -= np.max(A, axis = 1)[:, np.newaxis]
        A = np.exp(A)
        self.value = A / np.sum(A, axis = 1)[:, np.newaxis]
        return self.value
    
    def backprop(self):
        self.A.grad = self.value * (self.grad - np.sum(self.value * self.grad, axis = 1)[:, np.newaxis])
        self.A.backprop()

class CrossEntropy(Node):
    
    '''cross entropy loss without softmax built in
    assumes inputs are probabilities already, and the ground truth is one-hot encoded (or a dist)
    '''
    
    def __init__(self, yhat, y = None):
        self.value = None
        self.grad = 1
        self.yhat = yhat
        self.y = y
        
    def get_value(self):
        n = self.yhat.get_value().shape[0]
        self.value = -np.sum(self.y * np.log(self.yhat.value)) / n
        return self.value
    
    def backprop(self):
        self.yhat.grad = -self.y / (self.yhat.value.shape[0] * self.yhat.value)
        self.yhat.backprop()
        
class Relu(Node):
    
    '''rectified linear unit
    returns max(0, X) entry-wise
    '''
    
    def __init__(self, A):
        self.value = None
        self.grad = None
        self.A = A
    
    def get_value(self):
        self.value = np.maximum(0, self.A.get_value())
        return self.value
    
    def backprop(self):
        g = np.copy(self.grad)
        g[self.value == 0] = 0
        self.A.grad = g
        self.A.backprop()
        
def epoch(In, Out, Loss, weights, biases, batch_size, learning_rate, X_train, Y_train, X_val=None, Y_val=None, verbose=True):
    '''Performs one epoch of training.'''
    Y_train_one_hot = to_one_hot(Y_train)
    X_batches = np.split(X_train, np.arange(batch_size, X_train.shape[0], batch_size))
    Y_batches = np.split(Y_train_one_hot, np.arange(batch_size, Y_train.shape[0], batch_size))
    avg_loss = 0
    for i in range(len(X_batches)):
        In.value = X_batches[i]
        Loss.y = Y_batches[i]
        avg_loss += Loss.get_value()
        Loss.backprop()
        for weight in weights:
            weight.value -= learning_rate * weight.grad
        for bias in biases:
            bias.value -= learning_rate * bias.grad
    if verbose:
        print("Average training loss: %f" % (avg_loss / len(X_batches)))
        In.value = X_train
        predicted_class = np.argmax(Out.get_value(), axis = 1)
        print('Training accuracy: %.2f' % (np.mean(predicted_class == Y_train)))
        if X_val and Y_val:
            In.value = X_val
            predicted_class = np.argmax(Out.get_value(), axis = 1)
            print('Validation accuracy: %.2f' % (np.mean(predicted_class == Y_val)))
        
def to_one_hot(y):
    '''Converts a vector of 0,...,n-1 values to a one-hot encoded matrix.'''
    m = np.max(y) + 1
    z = np.zeros((y.shape[0], m))
    for i, elt in enumerate(y):
        z[i, elt] = 1
    return z