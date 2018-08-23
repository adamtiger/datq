import ae_train
from matplotlib import pyplot as plt

# this is for testing the modules
# later for managing the experiments
lr = 1e-3
iterations = 1
epochs = 1
outer_batch = 20
inner_batch = 1

_, history = ae_train.train_bae(lr, iterations, epochs, outer_batch, inner_batch)

plt.plot(history['iteration'], history['loss'], 'ro')
plt.show()
