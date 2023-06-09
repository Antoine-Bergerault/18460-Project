from client import Computation
from dataclasses import dataclass
from functools import cached_property
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from problem import OptimizationProblem
from tasks.task import Task, Config
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

default_config = Config(clients=[
    *(2*(Computation.HIGH,)),
    *(2*(Computation.LOW,))
], lr=0.001, nlr=0.004)#lambda k:0.006/sqrt(k)) #0.00148 0.00364575

solo_config = Config(clients=[
    Computation.HIGH
], lr=0.01, nlr=0.5)

class LogisticRegressionTask(Task):
    def __init__(self, config: Config = default_config) -> None:
        super().__init__(config)

        self.lr = config.lr
        self.clients = config.clients
        self.k = 1

    @cached_property
    def dataset(self):
        file_path = "data/uci-mushrooms/mushrooms.csv"
        col_names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 
                     'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring',
                     'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
        df = pd.read_csv(file_path, header=None, names=col_names)

        df = df.drop(index=[0]) # drop header

        for col in col_names:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])

        x = df.drop(columns=['class']) # leave only features
        y = df['class'] # classification

        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        x = np.hstack((x, np.ones((x.shape[0], 1)))) # add 1 for bias term

        dataset = np.column_stack((y, x))

        return dataset
        
    def get_subsets(self):
        return self.dataset.reshape((len(self.clients), -1, 23+1))

    def get_problem(self):
        hyper_parameters = {
            "penalty": 10,
            "x0": np.random.standard_normal((self.dataset[0, :].shape[0]-1, 1))
        }

        # simple linear regression cost (mean squared error loss)
        def cost(x, dataset, params):
            categories = dataset[:, 0]
            
            features = dataset[:, 1:]

            # sigmoid of -Aw
            prediction = 1/(1 + np.exp(-features @ x))
            
            return -categories.T @ np.log(prediction) - (1 - categories).T @ np.log(1 - prediction)

        def cost_grad(x, dataset, params):
            categories = dataset[:, 0, None]
            
            features = dataset[:, 1:]

            # sigmoid of -Aw
            prediction = 1/(1 + np.exp(-features @ x))

            grad = np.sum(features * (prediction - categories), axis=0)

            return grad[:, None] # make sure to return something with shape (2, 1)

        def cost_hessian(x, dataset, params):
            categories = dataset[:, 0]
            
            features = dataset[:, 1:]

            # sigmoid of -Aw
            prediction = 1/(1 + np.exp(-features @ x))
            hessian = features.T * np.diag(prediction * (1 - prediction)) @ features

            return hessian

        problem = OptimizationProblem(tol=1e-6, ctol=0.1, max_iter=20000, loss=cost, loss_grad=cost_grad, 
                                      loss_hessian=cost_hessian, hyper_parameters=hyper_parameters)
        
        return problem

    def visualize(self):
        pca = PCA(n_components=2)
        x_reduced = pca.fit_transform(self.dataset[:, 1:-1])
        y = self.dataset[:, 0]

        plt.scatter(x_reduced[y == 0, 0], x_reduced[y == 0, 1], color='red', label='Poisonous')
        plt.scatter(x_reduced[y == 1, 0], x_reduced[y == 1, 1], color='blue', label='Edible')
        plt.legend()
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA visualization of the mushroom dataset')
        plt.show()

    def visualize_solution(self, solution):
        pca = PCA(n_components=2)
        features = self.dataset[:, 1:]
        x_reduced = pca.fit_transform(features)
        y = self.dataset[:, 0]
        prediction = (1/(1 + np.exp(-features @ solution)) > 0.5).astype(int) # encode results

        fig = plt.gcf()
        fig.set_size_inches(12, 5)

        plt.subplot(1, 2, 1)
        plt.scatter(x_reduced[y == 0, 0], x_reduced[y == 0, 1], color='red', label='Poisonous')
        plt.scatter(x_reduced[y == 1, 0], x_reduced[y == 1, 1], color='blue', label='Edible')
        
        plt.legend()
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA visualization of logistic regression solution')

        plt.subplot(1, 2, 2)
        plt.scatter(x_reduced[prediction == 0, 0], x_reduced[prediction == 0, 1], color='red', label='Predicted Poisonous')
        plt.scatter(x_reduced[prediction == 1, 0], x_reduced[prediction == 1, 1], color='blue', label='Predicted Edible')

        plt.legend()
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA visualization of logistic regression prediction')

        plt.show()
