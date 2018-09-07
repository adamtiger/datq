import ae_train
from autoencoder import CNNSparseAE
import os
import datetime

def generate_folder(experiment_type):
    if experiment_type in ['ae', 'c', 'q']:
        dt = str(datetime.datetime.now())
        dt = dt.split('.')[0]
        dt = dt.replace('-', '')
        dt = dt.replace(':', '')
        dt = dt.replace(' ', '')
        dt = experiment_type + dt
        return dt
    else:
        return None

def create_folders(base_folder):
    base_folder = os.path.join('experiments', base_folder)
    os.mkdir(base_folder)

    weight_folder = os.path.join(base_folder, 'weights')
    os.mkdir(weight_folder)

    log_file = os.path.join(base_folder, 'logs.csv')
    return weight_folder, log_file


# managing the experiments
lr = 1e-3
iterations = 120
epochs = 5
outer_batch = 20000
inner_batch = 128
weight_folder, log_file = create_folders(generate_folder('ae'))
ae_train.file_name = log_file


ae_model = CNNSparseAE(0.2, 0.05)
_ = ae_train.train_ae(ae_model, weight_folder, lr, iterations, epochs, outer_batch, inner_batch, gpu_id=1, callback=ae_train.followup_performance)
#ae_train.plot_learning_curve("experiments/ae20180905181729/logs.csv")
#ae_train.plot_input_output(ae_model, path="experiments/ae20180905181729/weights/model_weights79.pt")