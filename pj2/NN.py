import six
import sys

sys.modules['sklearn.externals.six'] = six

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import mlrose_hiive as mlrose
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import time

def process_data():
#read dataset
    data = pd.read_csv('Italy_default_prior_2004.csv')
    data=data.sample(frac=1).reset_index(drop=True)#shuffle data
    data_x=data.loc[:,data.columns!="DefaultIndex"]
    le = LabelEncoder()
    for column in data_x.columns:
        data_x[column] = le.fit_transform(data_x[column])
    data_y = data.loc[:, data.columns == "DefaultIndex"]
    X = np.asarray(data_x)
    y = np.asarray(data_y.astype('int'))
    #split into train/test dataset
    x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=0)
    scaler = StandardScaler()
    scaled_x_train = scaler.fit_transform(x_train)
    scaled_x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, scaled_x_train, scaled_x_test



def run_NN(algorithm,scaled_x_train,scaled_x_test, y_train,y_test):
    results = []
    for i in range(1, 1002, 100):
    # for i in range(1, 5000, 500):
        if algorithm != "genetic_alg":
            model = mlrose.NeuralNetwork(hidden_nodes=[5,5], activation='relu',
                                     algorithm=algorithm, max_iters=i,
                                     bias=True, is_classifier=True, learning_rate=0.1,
                                     early_stopping=True, clip_max=5, max_attempts=100,
                                     random_state=10)
        else:
            model = mlrose.NeuralNetwork(hidden_nodes=[5,5], activation='relu',
                                     algorithm=algorithm, max_iters=int(i/10),
                                     bias=True, is_classifier=True, learning_rate=0.1,
                                     early_stopping=True, clip_max=5, max_attempts=100,
                                     random_state=10, pop_size=100, mutation_prob=0.1)
        start = time.time()
        model.fit(scaled_x_train, y_train)
        traintime = time.time() - start
        start = time.time()
        y_train_pred = model.predict(scaled_x_train)
        testtime = time.time() - start
        y_train_accuracy = accuracy_score(y_train, y_train_pred)

        y_test_pred = model.predict(scaled_x_test)
        y_test_accuracy = accuracy_score(y_test, y_test_pred)

        results.append([i, algorithm, y_train_accuracy, y_test_accuracy,traintime, testtime])
        print([i,algorithm])
    df = pd.DataFrame(results,columns=["Iterations", "Algorithm", "Train Accuracy", "Test Accuracy", "Train Time", "Test Time"])
    return df
def plot_result(df, algorithm):
    Accuracy_plot = sns.lineplot(data=df, x="Iterations", y="Test Accuracy", hue="Algorithm")
    # plt.savefig("results/Accuracy_plot_" + algorithm + " .png")
    plt.savefig("results/adhoc_Accuracy_plot_" + algorithm + " .png")
    plt.clf()
    traintime_plot = sns.lineplot(data=df, x="Iterations", y="Train Time", hue="Algorithm")
    # plt.savefig("results/traintime_plot_" + algorithm + " .png")
    plt.savefig("results/adhoc_traintime_plot_" + algorithm + " .png")
    plt.clf()

x_train, x_test, y_train, y_test, scaled_x_train, scaled_x_test = process_data()
# algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg']
algorithms = ['genetic_alg']

for algorithm in algorithms:
    df = run_NN(algorithm,scaled_x_train,scaled_x_test, y_train,y_test)
    plot_result(df, algorithm)

print('finish')