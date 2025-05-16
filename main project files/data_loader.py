import tensorflow as tf
import numpy as np

class DataLoader:
    def __init__(self, train_size=5000, test_size=10000):
        self.train_size = train_size
        self.test_size = test_size
        self.x_train = None
        self.y_train = None 
        self.x_test = None
        self.y_test = None
        self.load_data()
    
    def load_data(self):
        # Load and preprocess MNIST
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Normalization
        x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
        x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0
        
        # One-Hot-Encoding
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        # Training Set
        self.x_train = x_train[:self.train_size]
        self.y_train = y_train[:self.train_size]
        
        # Testing Set
        self.x_test = x_test[:self.test_size]
        self.y_test = y_test[:self.test_size]
    
    def get_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test
    
    def get_samples(self, num_samples=5):
        # Get random samples for visualization 
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        return self.x_test[indices], self.y_test[indices], indices