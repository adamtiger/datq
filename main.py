import argparse
import ae_train
from autoencoder import CNNSparseAE, load_model, numpy2torch_list
import clustering as cl
from sklearn.externals import joblib
import numpy as np
from q_learn import Q
import os
import csv
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
    params['lr'] = 5e-4
    params['iterations'] = 20
    params['epochs'] = 5
    params['outer_batch'] = 20000
    params['inner_batch'] = 128
    params['flat'] = False
    params['env'] = 'Pong-v0'
    weight_folder, log_file = create_folders(generate_folder('ae'))
    params['folder'] = weight_folder
    ae_train.file_name = log_file

    ae_model = CNNSparseAE(0.00, 0.05)
    _ = ae_train.train_ae(ae_model, params, callback=ae_train.followup_performance)
    #ae_train.plot_learning_curve("experiments/ae20180914110110/logs.csv")
    #ae_train.plot_input_output(
    #    ae_model,
    #    environment = params['env'],
    #    path="experiments/ae20180914110110/weights/model_weights9.pt"#os.path.join(weight_folder, ae_train._CONST_model_weights + str(params['iteration'] - 1) + '.pt')
    #)

# ---------------------------------
# Clustering
elif args.mode == 2:
    path = "experiments/ae20180914130809/weights/model_weights19.pt"
    sample_size = 5000
    num_clusters = 50
    batch_size = 256
    model_folder, log_file = create_folders(generate_folder('c'))
    
    # generate data from the environment
    print("Generate data.")
    ae_model = load_model(CNNSparseAE(), path)
    images, _ = ae_train.generate_samples(sample_size)
    tc_imgs = numpy2torch_list(images)
    tc_latents = list(map(ae_model.calculate_feature, tc_imgs))
    latents = np.concatenate(list(map(lambda x: x.detach().numpy(), tc_latents)))

    # cluster the data 
    clustering = cl.ClusteringKMeans(num_clusters, batch_size, latents)

    # saving results
    joblib.dump(clustering, os.path.join(model_folder, "clustering.pkl"))
    with open(log_file, 'wt', 1) as f:
        score = clustering.score(latents).tolist()
        writer = csv.writer(f)
        writer.writerow([score])

# ---------------------------------
# Q-learning
elif args.mode == 3:
    path_ae = "experiments/ae20180914130809/weights/model_weights19.pt"
    path_cl = "experiments/c20180917153834/weights/clustering.pkl"
    sample_size = 5000
    table_folder, log_file = create_folders(generate_folder('q'))
    env_name = 'Breakout-v0'
    
    # load the autoencoder model
    ae_model = load_model(CNNSparseAE(), path_ae)

    # read back the clustering model
    clustering = joblib.load(path_cl)

    # run the q-learning algorithm
    params = {}
    params['max_iter'] = 50
    params['gamma'] = 0.95
    params['epsilon_0'] = 0.9
    params['epsilon_min'] = 0.05
    params['epsilon_delta'] = (0.9 - 0.05) / 100.0
    params['eval_freq'] = 10
    
    f = open(log_file, 'at', 1)
    csv_writer = csv.writer(f)
    def save_records(returns):
        avg_ret = np.array(returns).mean()
        csv_writer.writerow([avg_ret])
    
    q = Q(ae_model, clustering, env_name, False, table_folder)
    q.train(params, callback=save_records)

    f.close()

print('Finished!')
