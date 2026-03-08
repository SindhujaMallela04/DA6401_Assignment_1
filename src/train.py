"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
import os
import json
import argparse
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from ann.neural_network import NeuralNetwork
from ann.activations import softmax

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('--dataset', type = str, default = 'mnist', choices = ['mnist', 'fashion_mnist'])
    parser.add_argument('--epochs', type = int, default = 5)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--learning_rate', type = float, default = 0.001)
    parser.add_argument('--optimizer', type = str, default = 'sgd', choices = ['sgd', 'momentum', 'nag', 'rmsprop'])
    parser.add_argument('--hidden_layers', type = str, default = "128, 64, 32, 16")
    parser.add_argument('--num_layers', type = int, default = None)
    parser.add_argument('--hidden_size', type = int, nargs = '+', default = None)
    parser.add_argument('--num_neurons', type = int, default = 128)
    parser.add_argument('--activation', type = str, default = 'relu', choices = ['relu', 'sigmoid', 'tanh'])
    parser.add_argument('--loss', type = str, default = 'cross_entropy', choices = ['cross_entropy', 'mse'])
    parser.add_argument('--weight_init', type = str, default = 'xavier', choices = ['xavier', 'random', 'zeros'])
    parser.add_argument('--wandb_project', type = str, default = 'DL_Assign_1_proj')
    parser.add_argument('--model_save_path', type = str, default = 'models/')
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()

    if args.hidden_size is not None:
        args.hidden_layers = args.hidden_size
    else :
        args.hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(",")]

    wandb.init(project = args.wandb_project, config = vars(args))

    if args.dataset == 'mnist' :
        from utils.data_loader import load_mnist_dataset, preprocess_data
        x_train, y_train, x_test, y_test = load_mnist_dataset()
        x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
        val_size = int(0.1 * x_train.shape[0])
        x_val, y_val = x_train[:val_size], y_train[:val_size]
        x_train, y_train = x_train[val_size:], y_train[val_size:]   
    elif args.dataset == 'fashion_mnist' :
        from utils.data_loader import load_fashion_mnist_dataset, preprocess_data
        x_train, y_train, x_test, y_test = load_fashion_mnist_dataset()
        x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
        val_size = int(0.1 * x_train.shape[0])
        x_val, y_val = x_train[:val_size], y_train[:val_size]
        x_train, y_train = x_train[val_size:], y_train[val_size:]
    else :
        raise ValueError("Invalid dataset choice")
    
    # table = wandb.Table(columns=["Class Name", "Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5"])
    # samples_per_class = 5
    # class_samples = {i: [] for i in range(10)}
    # for img, label in zip(x_train, y_train) :
    #     class_id = np.argmax(label)
    #     if len(class_samples[class_id]) < samples_per_class :
    #         class_samples[class_id].append(img.reshape(28, 28))
    #     if all(len(v) == samples_per_class for v in class_samples.values()) :
    #         break
    # for class_id in range(10) :
    #     images = [wandb.Image(img) for img in class_samples[class_id]]
    #     table.add_data(class_id, images[0], images[1], images[2], images[3], images[4])

    # wandb.log({"Section_2.1_MNIST_samples": table})

    model = NeuralNetwork(args)
    model.train(x_train, y_train, args.epochs, args.batch_size)

    train_accuracy = model.evaluate(x_train, y_train)
    val_accuracy = model.evaluate(x_val, y_val)
    test_accuracy = model.evaluate(x_test, y_test)

    # wandb.log({"train_accuracy" : train_accuracy, "val_accuracy" : val_accuracy, "test_accuracy" : test_accuracy})
    
    print(f"Train Accuracy: {train_accuracy: 4f}")
    print(f"Val Accuracy: {val_accuracy: 4f}")
    print(f"Test Accuracy: {test_accuracy: 4f}")
    print("Training complete!")

    # logits = model.forward(x_test)
    # y_pred = softmax(logits)

    # predictions = np.argmax(y_pred, axis = 1)
    # true_labels = np.argmax(y_test, axis = 1)

    # cm = confusion_matrix(true_labels, predictions)
    # plt.figure(figsize = (8, 6))
    # plt.imshow(cm, cmap="Blues")
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted Label")
    # plt.ylabel("True Label")
    # plt.colorbar()

    # ticks = np.arange(10)
    # plt.xticks(ticks, ticks)
    # plt.yticks(ticks, ticks)

    # plt.tight_layout()
    # plt.savefig("confusion_matrix.png")
    # plt.close()

    # wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})

    # misclassified = []
    # for img, pred, true in zip(x_test, predictions, true_labels) :
    #     if pred != true :
    #         misclassified.append((img, true, pred))
    #     if len(misclassified) == 10 :
    #         break

    # table = wandb.Table(columns = ["image", "true_label", "predicted_label"])
    # for img, true, pred in misclassified :
    #     table.add_data(wandb.Image(img.reshape(28, 28)), true, pred)

    # wandb.log({"misclassified_examples": table})


    best_weights = model.get_weights()
    np.save("best_model.npy", best_weights, allow_pickle = True)
    print("Model saved as best_model.npy")

    config = vars(args)
    with open("best_config.json", "w") as f :
        json.dump(config, f, indent = 4)
    print("Best config saved as best_config.json")


if __name__ == '__main__':
    main()

