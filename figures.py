from matplotlib import pyplot as plt

n_neurons = [5, 15, 50, 100, 250, 500]
sigmoid = [0.6853, 0.8199, 0.8347, 0.8378, 0.837, 0.8357]
arctan = [0.6198, 0.8885, 0.8682, 0.8811, 0.8471, 0.098]

plt.style.use('seaborn')
plt.title('Impact of number of neurons in hidden layer on accuracy')
plt.xlabel('Number of neurons')
plt.ylabel('Accuracy of neural network')
plt.scatter(n_neurons, sigmoid, label='sigmoid function', edgecolors='black', linewidths=0.5, s=100)
plt.scatter(n_neurons, arctan, label='arctan function', edgecolors='black', linewidths=0.5, s=100)
plt.legend()
plt.show()