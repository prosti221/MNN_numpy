import numpy as np

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
    sh = X.shape
    if len(sh) == 1:
        return np.concatenate([np.array([-1]), X])
    else:
        m = sh[0]
        bias = -np.ones((m,1)) # Makes a m*1 matrix of 1-s
        return np.concatenate([bias, X], axis  = 1) 

