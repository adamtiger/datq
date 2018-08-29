import ae_train
from autoencoder import CNNSparseAE
import os
import datetime

def generate_folder_name(experiment_type):
    if experiment_type in ['ae', 'c', 'q']:
        dt = str(datetime.datetime.now())
        dt = dt.split('.')[0]
        dt = dt.replace('-', '')
        dt = dt.replace(':', '')
        dt = dt.replace(' ', '')
        return experiment_type + dt
    else:
        return None

# managing the experiments
lr = 1e-3
iterations = 50
epochs = 2
outer_batch = 20
inner_batch = 4
folder = os.path.join('experiments', generate_folder_name('ae'))
ae_train.file_name = os.path.join(folder, 'logs.csv')


ae_model = CNNSparseAE(0.001, 0.05)

_ = ae_train.train_ae(ae_model, folder, lr, iterations, epochs, outer_batch, inner_batch, gpu_id=0, callback=ae_train.followup_performance)

ae_train.plot_learning_curve(ae_train.file_name)