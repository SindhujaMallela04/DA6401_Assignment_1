"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
import wandb
from .neural_layer import NeuralLayer
from .activations import relu, relu_derivative, softmax, tanh, tanh_derivative, sigmoid, sigmoid_derivative
from .objective_functions import cross_entropy_loss, cross_entropy_derivative, MSE_Loss, MSE_Loss_derivative
from .optimizers import SGD, Momentum, NAG, RMSProp

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        self.args = cli_args
        if hasattr(self.args, "hidden_layers") :
            hidden_layers = self.args.hidden_layers
        else :
            hidden_layers = self.args.hidden_size
        input_size = 784
        self.layers = []

        for size in hidden_layers :
            self.layers.append(NeuralLayer(input_size, size, self.args.weight_init))
            input_size = size

        self.layers.append(NeuralLayer(input_size, 10, self.args.weight_init))

        self.activation_funcs = []
        for _ in hidden_layers :
            if self.args.activation == 'relu' :
                self.activation_funcs.append(relu)
            elif self.args.activation == 'sigmoid' :
                self.activation_funcs.append(sigmoid)
            elif self.args.activation == 'tanh' :
                self.activation_funcs.append(tanh)
            else:
                raise ValueError("Invalid activation function")

        if self.args.loss == 'cross_entropy' :
            self.loss_func = cross_entropy_loss
            self.loss_grad = cross_entropy_derivative
        elif self.args.loss == 'mse' :
            self.loss_func = MSE_Loss
            self.loss_grad = MSE_Loss_derivative
        else :
            raise ValueError("Invalid loss function")

        if self.args.optimizer == 'sgd' :
            self.optimizer = SGD(self.args.learning_rate)
        elif self.args.optimizer == 'momentum' :
            self.optimizer = Momentum(self.args.learning_rate)
        elif self.args.optimizer == 'nag' :
            self.optimizer = NAG(self.args.learning_rate)
        elif self.args.optimizer == 'rmsprop' :
            self.optimizer = RMSProp(self.args.learning_rate)
        else :
            raise ValueError("Invalid optimizer")           
        

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """        

        for i in range(len(self.layers) - 1) :
            X = self.layers[i].forward(X)
            X = self.activation_funcs[i](X)
            # if i == 0:
            #     zero_fraction = np.mean(X == 0)
            #     wandb.log({"dead_neuron_fraction": zero_fraction})
        
        X = self.layers[-1].forward(X)

        self.logits = X
        return softmax(X), X

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        dZ = self.loss_grad(y_true, y_pred)
        # dZ = y_pred - y_true
        
        for i in reversed(range(len(self.layers))) :
            dX = self.layers[i].backward(dZ)
            if i > 0 :
                Z_prev = self.layers[i - 1].Z
                if self.args.activation == 'relu' :
                    dZ = dX * relu_derivative(Z_prev) 
                elif self.args.activation == 'sigmoid' :
                    dZ = dX * sigmoid_derivative(Z_prev)
                elif self.args.activation == 'tanh' :
                    dZ = dX * tanh_derivative(Z_prev)
            else :
                dZ = dX

        grad_W_list = []
        grad_b_list = []

        # Backprop through layers in reverse; collect grads so that index 0 = last layer
        for layer in reversed(self.layers) :
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)
        # create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)

        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        # print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
        # print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
        # layer_grads = self.layers[0].grad_W

        # for i in range(5) :
        #     neuron_grad = np.mean(np.abs(layer_grads[:, i]))
        #     wandb.log({f"neuron_grad_{i}": neuron_grad})

        return self.grad_W, self.grad_b
        

    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        for layer in self.layers :
            self.optimizer.update(layer)

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        """
        Train the network
        """
        num_samples = X_train.shape[0]

        for epoch in range(epochs) :
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            X_epoch = X_train[indices]
            y_epoch = y_train[indices]

            epoch_loss = 0

            for i in range (0, num_samples, batch_size) :
                X_batch = X_epoch[i:i + batch_size]
                y_batch = y_epoch[i:i + batch_size]   
                logits, y_pred = self.forward(X_batch)
                loss = self.loss_func(y_batch, y_pred) 
                epoch_loss += loss               
                self.backward(y_batch, y_pred)
                # grad_norm = np.linalg.norm(self.layers[0].grad_W)
                # wandb.log({"grad_norm_first_layer": grad_norm})
                self.update_weights()

            # wandb.log({"train_loss": epoch_loss})
        


    def evaluate(self, X, y):
        """
        Evaluate the model accuracy.
        """
        logits, y_pred = self.forward(X)
        predictions = np.argmax(y_pred, axis = 1)
        true_labels = np.argmax(y, axis = 1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

