import numpy as np

from Perceptron import Perceptron

training_inputs = []
training_inputs.append(np.array([0, 0, 0]))
training_inputs.append(np.array([0, 0, 1]))
training_inputs.append(np.array([0, 1, 0]))
training_inputs.append(np.array([0, 1, 1]))
training_inputs.append(np.array([1, 0, 0]))
training_inputs.append(np.array([1, 0, 1]))
training_inputs.append(np.array([1, 1, 0]))
training_inputs.append(np.array([1, 1, 1]))

labels = np.array([0, 1, 1, 1, 1, 1, 1, 1])

perceptron = Perceptron(3)
perceptron.train(training_inputs, labels)

inputs = np.array([1, 1, 1])
print("Predict for [1, 1, 1]: ",perceptron.predict(inputs))
#=> 1

inputs = np.array([0, 0, 0])
print("Predict for [0, 0, 0]: ", perceptron.predict(inputs))
#=> 0

inputs = np.array([1, 0, 0])
print("Predict for [1, 0, 0]: ", perceptron.predict(inputs))
#=> 1

inputs = np.array([0, 0, 1])
print("Predict for [0, 0, 1]: ", perceptron.predict(inputs))
#=> 1