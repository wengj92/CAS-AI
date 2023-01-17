import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):
    def __init__(self, dim_inputs, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.randn(dim_inputs + 1)

    def predict_batch(self, inputs):
        res_vector = np.dot(inputs, self.weights[1:]) + self.weights[0]
        activations = [1 if elem > 0 else 0 for elem in res_vector]
        return np.array(activations)

    def predict(self, inputs):
        res = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if res > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def get_loss(self, inputs, actual):
        # mae loss function
        return 1 / len(actual) * np.sum(np.abs(self.predict_batch(inputs) - actual))

    def plot_losses(self):
        plt.plot(self.training_loss, label='training loss')
        plt.plot(self.test_loss, label='test loss')
        plt.grid()
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('MAE loss')
        plt.show()

    def train(self, training_inputs, training_labels, test_inputs, test_labels):
        self.training_loss = []
        self.test_loss = []
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, training_labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
            # calculate loss
            self.training_loss.append(self.get_loss(training_inputs, training_labels))
            self.test_loss.append(self.get_loss(test_inputs, test_labels))
        # return test loss
        print(f'accuray on test-set is {100 - 100 * self.test_loss[-1]}%')



# EOF
