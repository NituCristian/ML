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
  return np.maximum(0, x)

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
    self.w7 = np.random.normal()
    self.w8 = np.random.normal()
    self.w9 = np.random.normal()
    self.w10 = np.random.normal()
    self.w11 = np.random.normal()
    self.w12 = np.random.normal()
    self.w13 = np.random.normal()
    self.w14 = np.random.normal()
    self.w15 = np.random.normal()
    self.w16 = np.random.normal()
    self.w17 = np.random.normal()
    self.w18 = np.random.normal()

    # Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()
    self.b4 = np.random.normal()
    self.b5 = np.random.normal()
    self.b6 = np.random.normal()
    self.b7 = np.random.normal()

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h1 = reLu(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    print("H1, W1, X[0], W2, X[1], b1", h1, self.w1, x[0], self.w2, x[1], self.b1)
    h2 = reLu(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    print("H2, W3, X[0], W4, X[1], b2", h2, self.w3, x[0], self.w4, x[1], self.b2)
    h3 = reLu(self.w5 * x[0] + self.w6 * x[1] + self.b3)
    print("H3, W5, X[0], W6, X[1], b3", h3, self.w5, x[0], self.w6, x[1], self.b3)
    h4 = sigmoid(self.w7 * h1 + self.w8 * h2 + self.w9 * h3 + self.b4)
    h5 = sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b5)
    h6 = sigmoid(self.w13 * h1 + self.w14 * h2 + self.w15 * h3 + self.b6)
    o1 = sigmoid(self.w16 * h4 + self.w17 * h5 + self.w18 * h6 + self.b7)
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

        sum_h3 = self.w5 * x[0] + self.w6 * x[1] + self.b3
        h3 = reLu(sum_h3)
############## better results if we apply sigmoid instead of relu on h4, h5, h6, so i preferred sigmoid ###########
        sum_h4 = self.w7 * h1 + self.w8 * h2 + self.w9 * h3 + self.b4
        h4 = sigmoid(sum_h4)

        sum_h5 = self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b5
        h5 = sigmoid(sum_h5)

        sum_h6 = self.w13 * h1 + self.w14 * h2 + self.w15 * h3 + self.b6
        h6 = sigmoid(sum_h6)

        sum_o1 = self.w16 * h4 + self.w17 * h5 + self.w18 * h6 + self.b7
        o1 = sigmoid(sum_o1)

        y_pred = o1

        # --- Calculate partial derivatives.
        # --- Naming: d_L_d_w1 represents "partial L / partial w1"
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Neuron o1
        d_ypred_d_w16 = h4 * deriv_sigmoid(sum_o1)
        d_ypred_d_w17 = h5 * deriv_sigmoid(sum_o1)
        d_ypred_d_w18 = h6 * deriv_sigmoid(sum_o1)
        d_ypred_d_b7 = deriv_sigmoid(sum_o1)

        d_ypred_d_h4 = self.w16 * deriv_sigmoid(sum_o1)
        d_ypred_d_h5 = self.w17 * deriv_sigmoid(sum_o1)
        d_ypred_d_h6 = self.w18 * deriv_sigmoid(sum_o1)

        # Neuron h1
        d_h1_d_w1 = x[0] * deriv_reLu(sum_h1)
        d_h1_d_w2 = x[1] * deriv_reLu(sum_h1)
        d_h1_d_b1 = deriv_reLu(sum_h1)

        # Neuron h2
        d_h2_d_w3 = x[0] * deriv_reLu(sum_h2)
        d_h2_d_w4 = x[1] * deriv_reLu(sum_h2)
        d_h2_d_b2 = deriv_reLu(sum_h2)

        # Neuron h3
        d_h3_d_w5 = x[0] * deriv_reLu(sum_h3)
        d_h3_d_w6 = x[1] * deriv_reLu(sum_h3)
        d_h3_d_b3 = deriv_reLu(sum_h3)

        # Neuron h4
        d_h4_d_w7 = h1 * deriv_sigmoid(sum_h4)
        d_h4_d_w8 = h2 * deriv_sigmoid(sum_h4)
        d_h4_d_w9 = h3 * deriv_sigmoid(sum_h4)
        d_h4_d_b4 = deriv_sigmoid(sum_h4)

        d_h4_d_h1 = self.w7 * deriv_sigmoid(sum_h4)
        d_h4_d_h2 = self.w8 * deriv_sigmoid(sum_h4)
        d_h4_d_h3 = self.w9 * deriv_sigmoid(sum_h4)

        # Neuron h5
        d_h5_d_w10 = h1 * deriv_sigmoid(sum_h5)
        d_h5_d_w11 = h2 * deriv_sigmoid(sum_h5)
        d_h5_d_w12 = h3 * deriv_sigmoid(sum_h5)
        d_h5_d_b5 = deriv_sigmoid(sum_h5)

        d_h5_d_h1 = self.w10 * deriv_sigmoid(sum_h5)
        d_h5_d_h2 = self.w11 * deriv_sigmoid(sum_h5)
        d_h5_d_h3 = self.w12 * deriv_sigmoid(sum_h5)

        # Neuron h6
        d_h6_d_w13 = h1 * deriv_sigmoid(sum_h6)
        d_h6_d_w14 = h2 * deriv_sigmoid(sum_h6)
        d_h6_d_w15 = h3 * deriv_sigmoid(sum_h6)
        d_h6_d_b6 = deriv_sigmoid(sum_h6)

        d_h6_d_h1 = self.w13 * deriv_sigmoid(sum_h6)
        d_h6_d_h2 = self.w14 * deriv_sigmoid(sum_h6)
        d_h6_d_h3 = self.w15 * deriv_sigmoid(sum_h6)

        # --- Update weights and biases
        # Neuron h1
        self.w1 -= learn_rate * d_L_d_ypred * (d_ypred_d_h4 * d_h4_d_h1 + d_ypred_d_h5 * d_h5_d_h1 + d_ypred_d_h6 * d_h6_d_h1) * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * (d_ypred_d_h4 * d_h4_d_h1 + d_ypred_d_h5 * d_h5_d_h1 + d_ypred_d_h6 * d_h6_d_h1) * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * (d_ypred_d_h4 * d_h4_d_h1 + d_ypred_d_h5 * d_h5_d_h1 + d_ypred_d_h6 * d_h6_d_h1) * d_h1_d_b1

        # Neuron h2
        self.w3 -= learn_rate * d_L_d_ypred * (d_ypred_d_h4 * d_h4_d_h2 + d_ypred_d_h5 * d_h5_d_h2 + d_ypred_d_h6 * d_h6_d_h2) * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * (d_ypred_d_h4 * d_h4_d_h2 + d_ypred_d_h5 * d_h5_d_h2 + d_ypred_d_h6 * d_h6_d_h2) * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * (d_ypred_d_h4 * d_h4_d_h2 + d_ypred_d_h5 * d_h5_d_h2 + d_ypred_d_h6 * d_h6_d_h2) * d_h2_d_b2

        # Neuron h3
        self.w5 -= learn_rate * d_L_d_ypred * (d_ypred_d_h4 * d_h4_d_h3 + d_ypred_d_h5 * d_h5_d_h3 + d_ypred_d_h6 * d_h6_d_h3) * d_h3_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * (d_ypred_d_h4 * d_h4_d_h3 + d_ypred_d_h5 * d_h5_d_h3 + d_ypred_d_h6 * d_h6_d_h3) * d_h3_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * (d_ypred_d_h4 * d_h4_d_h3 + d_ypred_d_h5 * d_h5_d_h3 + d_ypred_d_h6 * d_h6_d_h3) * d_h3_d_b3

        # Neuron h4
        self.w7 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w7
        self.w8 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w8
        self.w9 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w9
        self.b4 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_b4

        # Neuron h5
        self.w10 -= learn_rate * d_L_d_ypred * d_ypred_d_h5 * d_h5_d_w10
        self.w11 -= learn_rate * d_L_d_ypred * d_ypred_d_h5 * d_h5_d_w11
        self.w12 -= learn_rate * d_L_d_ypred * d_ypred_d_h5 * d_h5_d_w12
        self.b5 -= learn_rate * d_L_d_ypred * d_ypred_d_h5 * d_h5_d_b5

        # Neuron h6
        self.w13 -= learn_rate * d_L_d_ypred * d_ypred_d_h6 * d_h6_d_w13
        self.w14 -= learn_rate * d_L_d_ypred * d_ypred_d_h6 * d_h6_d_w14
        self.w15 -= learn_rate * d_L_d_ypred * d_ypred_d_h6 * d_h6_d_w15
        self.b6 -= learn_rate * d_L_d_ypred * d_ypred_d_h6 * d_h6_d_b6

        # Neuron o1
        self.w16 -= learn_rate * d_L_d_ypred * d_ypred_d_w16
        self.w17 -= learn_rate * d_L_d_ypred * d_ypred_d_w17
        self.w18 -= learn_rate * d_L_d_ypred * d_ypred_d_w18
        self.b7 -= learn_rate * d_L_d_ypred * d_ypred_d_b7

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

weightBoys = np.round(np.random.normal(80, 5, 2), 0)
weightBoys[0:] -= 63
print("Weight girls, weight boys", weightGirls, weightBoys)

#default height: 167 cm
heightGirls = np.round(np.random.normal(160, 10, 2), 0) # -167cm
heightGirls[0:] -= 167

heightBoys = np.round(np.random.normal(180, 10, 2), 0) # -167cm
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

emily = np.array([-7, -3])
frank = np.array([20, 15])

print("Emily: %.3f" % network.feedforward(emily))
print("Frank: %.3f" % network.feedforward(frank))