import argparse
import ae_train
from autoencoder import CNNSparseAE
import os
import datetime

parser = argparse.ArgumentParser(description="Manage training phase.")
parser.add_argument("--mode", type=int, default=0, metavar='N', help="1: ae, 2: c, 3: q")
args = parser.parse_args()

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


# ---------------------------------
# Autoencoder training
if args.mode == 1:

    # training parameters
    params = {}
    params['gpu_id'] = 0
    params['lr'] = 1e-3
    params['iterations'] = 120
    params['epochs'] = 5
    params['outer_batch'] = 20000
    params['inner_batch'] = 128
    weight_folder, log_file = create_folders(generate_folder('ae'))
    params['folder'] = weight_folder
    ae_train.file_name = log_file

    ae_model = CNNSparseAE(0.2, 0.05)
    _ = ae_train.train_ae(ae_model, params, callback=ae_train.followup_performance)
    ae_train.plot_learning_curve(log_file)
    ae_train.plot_input_output(
        ae_model, 
        path=os.path.join(weight_folder, ae_train._CONST_model_weights + str(params['iteration'] - 1) + '.pt')
    )

# ---------------------------------
# Clustering
elif args.mode == 2:
    pass

# ---------------------------------
# Q-learning
elif args.mode == 3:
    pass


print('Finished!')
