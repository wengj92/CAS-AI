import numpy as np
import matplotlib.pyplot as plt
import logging


class GradientDescent:
    def __init__(self):
        self.rest_logs()

    def rest_logs(self):
        self.history_m = []
        self.history_b = []
        self.losses = []

    def update_weights(self, m, b, x, y, learning_rate):
        m_deriv = 0
        b_deriv = 0
        # calculate derivatives
        for xi, yi in zip(x, y):
            m_deriv += -2 * np.exp(2 * xi) * (yi - (m * np.exp(2 * xi) + b))
            b_deriv += -2 * ((yi - (m * np.exp(2 * xi) + b)))
        m_deriv = m_deriv / float(len(x))
        b_deriv = b_deriv / float(len(x))
        # We subtract because the derivatives point in direction of steepest ascent
        m -= m_deriv * learning_rate
        b -= b_deriv * learning_rate
        # append results to history log
        self.history_m.append(m)
        self.history_b.append(b)
        # return results
        return m, b

    def train(self, x, y, epochs=1000, learning_rate=0.001):
        m = 0
        b = 0
        self.epochs = epochs
        self.rest_logs()
        for i in range(self.epochs):
            # calculate updated weights
            m, b = self.update_weights(m, b, x, y, learning_rate)
            # calculate difference between exact and approximate solution
            diffs = (y - (m * np.exp(2 * x) + b)) ** 2
            # calculate loss
            loss = np.sum(diffs) / len(x)
            self.losses.append(loss)
            if i % 500 == 0:
                logging.info(msg=f'{loss} ({m}, {b})')
        return m, b

    def plot_history(self, m, b):
        plt.plot(self.history_m[0:self.epochs])
        plt.plot(self.history_b[0:self.epochs])
        plt.axhline(y=m, xmin=0, xmax=self.epochs, c='r', linestyle='--')
        plt.axhline(y=b, xmin=0, xmax=self.epochs, c='b', linestyle=':')
        plt.grid()
        plt.xlabel('epoch')
        plt.legend(['m', 'b'], loc='upper right')
        plt.show()

    def plot_losses(self):
        plt.plot(self.losses)
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()


# EOF
