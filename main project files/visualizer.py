import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd

class Visualizer:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
    
    def display_results_table(self, results):
        """Display a table with performance metrics"""
        table_data = {
            'Metric': ['Best Loss', 'Final Loss', 'Final Test Accuracy', 'Total Iterations', 'Execution Time (s)'],
            'Value': [
                f"{results['best_loss']:.4f}",
                f"{results['final_loss']:.4f}",
                f"{results['final_accuracy']:.4f}",
                f"{results['total_iterations']}",
                f"{results['execution_time']:.2f}"
            ]
        }
        
        print("\nPerformance Summary:")
        print("-" * 40)
        for i in range(len(table_data['Metric'])):
            print(f"{table_data['Metric'][i]:<20}: {table_data['Value'][i]}")
        print("-" * 40)
        
        return table_data
    
    def plot_training_progress(self, history):
        """Plot the training progress (loss and accuracy)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss over Iterations')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(history['accuracy'], label='Training Accuracy')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Accuracy over Iterations')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self):
        """Plot the confusion matrix for the test data"""
        _, _, x_test, y_test = self.data_loader.get_data()
        y_pred = self.model.predict(x_test)
        y_true = np.argmax(y_test, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def display_classification_report(self):
        """Display the classification report"""
        _, _, x_test, y_test = self.data_loader.get_data()
        y_pred = self.model.predict(x_test)
        y_true = np.argmax(y_test, axis=1)
        
        report = classification_report(y_true, y_pred, digits=4)
        
        # Create a DataFrame for visualization
        report_data = classification_report(y_true, y_pred, digits=4, output_dict=True)
        df_report = pd.DataFrame(report_data).transpose()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_report.iloc[:-3, :].astype(float), annot=True, cmap='Blues')
        plt.title('Classification Report')
        plt.tight_layout()
        plt.show()

    def display_samples(self, num_samples=5):
        """Display sample images with actual and predicted labels"""
        x_samples, y_samples, indices = self.data_loader.get_samples(num_samples)
        y_true = np.argmax(y_samples, axis=1)
        y_pred = self.model.predict(x_samples)
        
        plt.figure(figsize=(15, 3))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(x_samples[i].reshape(28, 28), cmap='gray')
            plt.title(f"True: {y_true[i]}, Pred: {y_pred[i]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_all(self, results):
        """Run all visualization methods"""
        self.display_results_table(results)
        self.plot_training_progress(results['history'])
        self.plot_confusion_matrix()
        self.display_classification_report()
        self.display_samples()