import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self, input_dim=784, hidden_dim=32, output_dim=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.vector_size = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize the model’s parameters inside the class, so that they can be used during forward or backward passes.
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2 / self.input_dim)
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * np.sqrt(2 / self.hidden_dim)
        self.b2 = np.zeros(self.output_dim)
    
    def he_vector(self):
        """Generate a vector using He initialization"""
        # Generate a vector representing a possible solution (for optimization algorithms like Genetic Algorithms, Differential Evolution, etc.)
        w1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2 / self.input_dim)
        b1 = np.zeros(self.hidden_dim)
        w2 = np.random.randn(self.hidden_dim, self.output_dim) * np.sqrt(2 / self.hidden_dim)
        b2 = np.zeros(self.output_dim)
        return np.concatenate([w1.flatten(), b1, w2.flatten(), b2])
    
    def decode_vector(self, vector):
        """Decode a flat vector into the network weights"""
        W1 = vector[0:self.input_dim * self.hidden_dim].reshape((self.input_dim, self.hidden_dim))
        b1 = vector[self.input_dim * self.hidden_dim:self.input_dim * self.hidden_dim + self.hidden_dim].reshape((self.hidden_dim,))
        W2 = vector[self.input_dim * self.hidden_dim + self.hidden_dim:self.input_dim * self.hidden_dim + self.hidden_dim + self.hidden_dim * self.output_dim].reshape((self.hidden_dim, self.output_dim))
        b2 = vector[self.input_dim * self.hidden_dim + self.hidden_dim + self.hidden_dim * self.output_dim:].reshape((self.output_dim,))
        return W1, b1, W2, b2
    
    def set_weights_from_vector(self, vector):
        """Set the network weights from a flat vector"""
        self.W1, self.b1, self.W2, self.b2 = self.decode_vector(vector)
    
    def get_weights_as_vector(self):
        """Convert the network weights to a flat vector"""
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def softmax(self, x):
        """Softmax activation function with numerical stability"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward pass through the network (for backpropagation)"""  
        Z1 = X @ self.W1 + self.b1      # Linear part
        A1 = self.relu(Z1)              # Apply ReLU
        Z2 = A1 @ self.W2 + self.b2     # Linear part
        A2 = self.softmax(Z2)           # Apply softmax
        return A2, A1, Z1
    
    def forward_from_vector(self, X, vector):
        """Forward pass using weights from a vector (for EA)"""
        W1, b1, W2, b2 = self.decode_vector(vector)
        Z1 = X @ W1 + b1      # Linear part
        A1 = self.relu(Z1)    # Apply ReLU
        Z2 = A1 @ W2 + b2     # Linear part
        A2 = self.softmax(Z2) # Apply softmax
        return A2
    
    def cross_entropy(self, preds, targets, eps=1e-12):
        """Compute cross-entropy loss"""
        preds = np.clip(preds, eps, 1 - eps)  # Avoid log(0) and log(1)
        return -np.mean(np.sum(targets * np.log(preds), axis=1))
    
    def predict(self, X):
        """Make predictions for input data"""
        probs, _, _ = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        probs, _, _ = self.forward(X)
        loss = self.cross_entropy(probs, y)
        y_pred = np.argmax(probs, axis=1)
        y_true = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred == y_true)
        return loss, accuracy
    
    def fitness_function(self, vector, x_data, y_data):
        """Compute the fitness (loss) for a given weight vector"""
        probs = self.forward_from_vector(x_data, vector)
        ce_loss = self.cross_entropy(probs, y_data)
        return ce_loss
    
    def save_model(self, filename):
        """Save the model weights to a file"""
        model_data = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filename):
        """Load a model from a file"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            input_dim=model_data['input_dim'],
            hidden_dim=model_data['hidden_dim'],
            output_dim=model_data['output_dim']
        )
        
        model.W1 = model_data['W1']
        model.b1 = model_data['b1']
        model.W2 = model_data['W2']
        model.b2 = model_data['b2']
        
        return model