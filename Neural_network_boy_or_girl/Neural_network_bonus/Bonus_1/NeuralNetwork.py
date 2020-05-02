import numpy as np


def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def reLu(x):
  # ReLU activation function: f(x) = max(0, x)
  if (x < 0):
    return 0
  return x


def deriv_reLu(x):
  # Derivative of ReLU: f'(x) = 0 if x < 0; 1 if x >= 0
  if(x < 0):
    return 0
  return 1

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)

  *** DISCLAIMER ***:
  The code below is intended to be simple and educational, NOT optimal.
  Real neural net code looks nothing like this. DO NOT use this code.
  Instead, read/run it to understand how this specific network works.
  '''
  def __init__(self):
    # Weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h1 = reLu(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    print("H1, W1, X[0], W2, X[1], b1", h1, self.w1, x[0], self.w2, x[1], self.b1)
    h2 = reLu(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    print("H2, W3, X[0], W4, X[1], b2", h2, self.w3, x[0], self.w4, x[1], self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    print("O1, W5, H1,  W6, H2, b3", o1, self.w5, h1, self.w6, h2, self.b3)
    return o1

  def train(self, data, all_y_trues):
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    '''
    learn_rate = 0.1
    epochs = 1000 # number of times to loop through the entire dataset

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- Do a feedforward (we'll need these values later)
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = reLu(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = reLu(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # --- Calculate partial derivatives.
        # --- Naming: d_L_d_w1 represents "partial L / partial w1"
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Neuron o1
        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_b3 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

        # Neuron h1
        d_h1_d_w1 = x[0] * deriv_reLu(sum_h1)
        d_h1_d_w2 = x[1] * deriv_reLu(sum_h1)
        d_h1_d_b1 = deriv_reLu(sum_h1)

        # Neuron h2
        d_h2_d_w3 = x[0] * deriv_reLu(sum_h2)
        d_h2_d_w4 = x[1] * deriv_reLu(sum_h2)
        d_h2_d_b2 = deriv_reLu(sum_h2)

        # --- Update weights and biases
        # Neuron h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Neuron h2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Neuron o1
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

      # --- Calculate total loss at the end of each epoch
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        print("Y preds", y_preds)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))


# Define dataset

#default weight: 63 kg
weightGirls = np.round(np.random.normal(60, 5, 2), 0)
weightGirls[0:] -= 63

weightBoys = np.round(np.random.normal(85, 5, 2), 0)
weightBoys[0:] -= 63
print("Weight girls, weight boys", weightGirls, weightBoys)

#default height: 167 cm
heightGirls = np.round(np.random.normal(160, 10, 2), 0) # -167cm
heightGirls[0:] -= 167

heightBoys = np.round(np.random.normal(185, 10, 2), 0) # -167cm
heightBoys[0:] -= 167
print("Height girls, height boys", heightGirls, heightBoys)



data = np.array([
  [weightGirls[0], heightGirls[0]],  # Alice
  [weightBoys[0], heightBoys[0]],   # Bob
  [weightBoys[1], heightBoys[1]],   # Charlie
  [weightGirls[1], heightGirls[1]], # Diana
])


print("Data", data)


all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1,  # Diana
])

# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)

print("Weight girls, weight boys", weightGirls, weightBoys)
print("Height girls, height boys", heightGirls, heightBoys)

emily = np.array([-7, -3])
frank = np.array([15, 15])

print("Emily: %.3f" % network.feedforward(emily))
print("Frank: %.3f" % network.feedforward(frank))