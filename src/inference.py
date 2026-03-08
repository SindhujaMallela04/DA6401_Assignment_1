"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from ann.neural_network import NeuralNetwork
from ann.activations import softmax
from utils.data_loader import load_mnist_dataset, load_fashion_mnist_dataset, preprocess_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('--model_path', type = str, default = 'best_model.npy')
    parser.add_argument('--dataset', type = str, default = 'mnist', choices = ['mnist', 'fashion_mnist'])
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--learning_rate', type = float, default = 0.001)
    parser.add_argument('--optimizer', type = str, default = 'sgd', choices = ['sgd', 'momentum', 'nag', 'rmsprop'])
    parser.add_argument('--hidden_layers', type = str, default = "128, 64, 32, 16")
    parser.add_argument('--activation', type = str, default = 'relu', choices = ['relu', 'sigmoid', 'tanh'])
    parser.add_argument('--loss', type = str, default = 'cross_entropy', choices = ['cross_entropy', 'mse'])
    parser.add_argument('--weight_init', type = str, default = 'xavier', choices= ['xavier', 'random'])
        
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    weights = np.load(model_path, allow_pickle = True).item()
    return weights


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    probs, logits = model.forward(X_test)
    y_pred = np.argmax(probs, axis = 1)
    y_true = np.argmax(y_test, axis = 1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average = 'macro')
    recall = recall_score(y_true, y_pred, average = 'macro')
    f1 = f1_score(y_true, y_pred, average = 'macro')

    results = {"logits": logits, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    return results


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    args.hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(",")]

    if args.dataset == 'mnist' :
        x_train, y_train, x_test, y_test = load_mnist_dataset()
    else :
        x_train, y_train, x_test, y_test = load_fashion_mnist_dataset()

    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)

    model = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    results = evaluate_model(model, x_test, y_test)

    print("Accuracy :", results["accuracy"])
    print("Precision :", results["precision"])
    print("Recall :", results["recall"])
    print("F1-Score :", results["f1"])
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
