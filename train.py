import numpy as np
from keras.datasets import mnist

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

