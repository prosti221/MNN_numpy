import numpy as np
from keras.datasets import mnist


class MNNClassifier():
    """A multi-layer neural network with one hidden layer"""
    
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
            
    def forward(self, X):
        a_in = X @ self.weights1
        a_out = self.sigmoid(a_in)
        y_in = a_out_bias = self.add_bias(a_out)
        y_in = y_in @ self.weights2
        y_out = self.sigmoid(y_in)
        return (a_out_bias, y_out)
        
    def predict(self, X_train):
        return np.argmax(self.forward(self.add_bias(self.scale(X_train)))[1], axis=1)
    
    def score(self, X_train):
        return self.forward(self.add_bias(self.scale(X_train)))[1][:, 1]
    
    def accuracy(self, X_train, t_test, y_out=None):
        if y_out is not None:
            acc = [int(np.argmax(y_out[i]) == t_test[i]) for i in range(t_test.shape[0])]
        else:
            inp = self.add_bias(self.scale(X_train))
            acc = [int(np.argmax(self.forward(inp[i])[1]) == t_test[i]) for i in range(t_test.shape[0])]
        return sum(acc) / len(acc)
        
    def scale(self, data_set): #Using standard scaling
        data_points, feature_n = data_set.shape;
        #Standard deviation and mean for each feature
        std_i = [np.std(data_set[:,i]) for i in range(feature_n)]
        mean_i = [np.average(data_set[:,i]) for i in range(feature_n)]
        #Normalizing each feature set of X_train
        feature_i_norm = [(data_set[:, i] - mean_i[i]) / std_i[i] for i in range(feature_n)]
        data_set_norm = [[feature_i_norm[i][j] for i in range(feature_n)] 
                                               for j in range(data_points)]
        return np.array(data_set_norm)
    
    def encode(self, data_set): #Converts output to one-hot encoding
        class_n = np.max(data_set) + 1; data_points = len(data_set)
        return np.array([[np.array((data_set[i] == j)).astype('int') for j in range(class_n)] 
                                                                     for i in range(data_points)]) 
    def init_weights(self, dim_in, dim_out, dim_hidden):
        w1 = np.random.randn(dim_in + 1, dim_hidden)/np.sqrt(dim_in+1 * dim_hidden)
        w2 = np.random.randn(dim_hidden + 1, dim_out)/np.sqrt(dim_hidden+1 * dim_out)
        return (w1, w2)
    
    def sigmoid(self, X):
        return 1.0/(1.0+np.exp(-X))
    
    def add_bias(self, X):
        # Put bias in position 0
        sh = X.shape
        if len(sh) == 1:
            #X is a vector
            return np.concatenate([np.array([-1]), X])
        else:
            # X is a matrix
            m = sh[0]
            bias = -np.ones((m,1)) # Makes a m*1 matrix of 1-s
            return np.concatenate([bias, X], axis  = 1) 

if __name__ == '__main__':
    (X_train, t_train), (X_test, t_test) = mnist.load_data()
    
    #Reshape
    train_X = m_train_X.reshape(m_train_X.shape[0], 784)
    test_X = m_train_X.reshape(m_train_X.shape[0], 784)

    runs = [(MNNClassifier(eta = 0.009, dim_hidden = 42)) for i in range(10)]
    for run in runs:
        run.fit(X_train, t_train, validation_set=(X_val, t_val))
        
    best_accuracies = [run.best_accuracy for run in runs] #Best accuracies on X_val, t_val
    print('Mean accuracy: %f\n' %np.average(best_accuracies))
    print('Standard deviation: %f\n' %np.std(best_accuracies))
    best_multi = runs[np.argmax(best_accuracies)] #Plotting the best run
