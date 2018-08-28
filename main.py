import ae_train
from autoencoder import CNNSparseAE

# this is for testing the modules
# later for managing the experiments
lr = 1e-3
iterations = 2
epochs = 2
outer_batch = 20
inner_batch = 4
ae_train.file_name = 'logs.csv'


ae_model = CNNSparseAE(0.001, 0.05)

_ = ae_train.train_ae(ae_model, lr, iterations, epochs, outer_batch, inner_batch, callback=ae_train.followup_performance)

ae_train.plot_learning_curve(ae_train.file_name)