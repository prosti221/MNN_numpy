import numpy as np
from utils import *
from keras.datasets import mnist


class MNNClassifier():
    def __init__(self,eta = 0.0089, dim_hidden = 42): #eta = 0.0096, hidden=42 best so far
        self.eta = eta
        self.dim_hidden = dim_hidden
        self.weights1, self.weights2 = (None, None)
        self.accuracies = []    #list of accuracies for the validation data
        self.best_accuracy = -1 #Highest accuracy during training for the validation data
                
    def fit(self, X_train, t_train, epochs = 2500, validation_set=None):
        # Initilaization
        dim_in =  X_train.shape[1]
        dim_out = self.encode(t_train).shape[1]
        self.weights1, self.weights2 = self.init_weights(dim_in, dim_out, self.dim_hidden)
        t_train_enc = self.encode(t_train)
        inp_scaled = self.add_bias(self.scale(X_train))
        validation_set = (X_train, t_train) if validation_set is None else validation_set
        #Array with accuracy and copy of the best weights so far for the validation data
        best_weights = [-1, None]
        for e in range(epochs):
            #Run the forward pass
            a_out, y_out = self.forward(inp_scaled)
            #Check accuracy
            if ((acc := self.accuracy(*validation_set, y_out=None)) > best_weights[0]):
                best_weights = [acc, (np.copy(self.weights1), np.copy(self.weights2))]
                self.best_accuracy = acc
            self.accuracies.append(acc)
            #delta terms
            delta_0 = (y_out - t_train_enc) * y_out * (1 - y_out)
            delta_1 = a_out*(1 - a_out) * np.dot(delta_0, self.weights2.T)
            #Update weights
            self.weights2 -= self.eta * np.dot(a_out.T, delta_0) 
            self.weights1 -= self.eta * np.dot(inp_scaled.T, delta_1[: , :-1])
        #Assign the weights to the ones with the best accuracy
        self.weights1, self.weights2 = best_weights[1]
