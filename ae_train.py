import os
import gym
import csv
import torch
import numpy as np
import pandas as pd
from losses import sparsity_loss
from skimage.util import crop
from skimage.color import rgb2grey
from skimage.transform import resize
from autoencoder import TrainLoader, get_device, Train, CNNControlAE
from environment import Environment as E
from matplotlib import pyplot as plt


# Fixed names for files.
_CONST_model_weights = "model_weights"
_CONST_learning_curve = "lrn.png"
_CONST_input = "input.png"
_CONST_output = "output.png"

def generate_samples(batch_size, flat=False, environment='Breakout-v0'):
    '''
    batch_size - the number of images to generate during a run
    environment - name of the OpenAI like environment
    '''
    # create the environment and reset
    env = gym.make(environment)

    # generate images
    images, dones =  E.generate_random_trajectory(batch_size, env)
    images = E.preprocess_batch(images)
    images, dones = E.concatenate(images, dones)
    if flat:
        images = E.flatten(images)
    
    return images, dones

def train_ae(ae_model, params, callback=None):
    '''
    The training function for AutoEncoders. This is a general function.
    callback - function for handling the administrative data
    '''
    device = get_device(cpu_anyway=(params['gpu_id']==-1), gpu_id=params['gpu_id'])
    
    for i in range(params['iterations']):
        print("Iteration (starting): [%d]"%i)

        # create a batch for the outer loop
        images, _ = generate_samples(params['outer_batch'], params['flat'], params['env']) # random_image(outer_batch)
        trainloader = TrainLoader(images, params['inner_batch']).get_trainloader()
        
        trainer = Train(ae_model, device, params['epochs'], params['lr'])
        trainer.fit(trainloader, callback=callback)
        if (i+1) % 10 == 0:
            trainer.save(os.path.join(params['folder'], _CONST_model_weights + str(i) + ".pt"))
    
    return ae_model


file_name = "logs.csv"
history = {'iteration':[], 'loss_reg':[], 'loss_rec':[]}
itr = 0
def followup_performance(epoch, i, loss_rec_item, loss_reg_item, x_size):
    global itr

    history['iteration'].append(itr)
    history['loss_rec'].append(loss_rec_item)
    history['loss_reg'].append(loss_reg_item)

    if i == 0:
        c_write = open(file_name, 'at', buffering=1)
        c_read = open(file_name, 'rt')

        csv_reader = csv.reader(c_read)
        csv_writer = csv.writer(c_write)
        if len(list(csv_reader)) == 0:
            # write the headers into the file
            csv_writer.writerow(history.keys())
        else:
            for idx in range(len(history['iteration'])):
                row = []
                for k in history.keys():
                    row.append(history[k][idx])
                csv_writer.writerow(row)
            for k in history.keys():
                history[k].clear()
        
        c_write.close()
        c_read.close()

    itr += 1

def plot_learning_curve(file, verbose=False):
    '''
    Plots the total loss at each iteration.
    file - the csv file containing (iteration, loss_rec, loss_reg)
    '''
    df = pd.read_csv(file)
    x = df['iteration'].values
    y_rec = df['loss_rec']
    y_reg = df['loss_reg']
    plt.plot(x, y_rec, 'yo', x, y_reg, 'ro')
    file = file.replace(file.split(os.sep)[-1], _CONST_learning_curve)
    plt.savefig(file)
    if verbose:
        plt.show()

def plot_input_output(ae_model, environment='Breakout-v0', path=None, verbose=False):
    
    # parent folder of parent folder
    def pp_folder(path, new_name):
        items = path.split(os.sep)
        old = os.path.join(items[-2], items[-1])
        new = new_name
        path = path.replace(old, new)
        return path
    
    if path is not None:
        ae_model.load_state_dict(torch.load(path, map_location='cpu'))
    imgs, _ = generate_samples(20, environment=environment)
    img = torch.tensor(imgs[0], dtype=torch.float32).view(1, 4, 108, 84)

    y = ae_model(img)
    
    plt.figure(8)
    plt.imshow(img.detach().numpy()[0, 2]*255)
    if path is not None:
        plt.savefig(pp_folder(path, _CONST_input))
    if verbose:
        plt.show()
    
    plt.figure(9)
    plt.imshow(y.detach().numpy()[0, 2]*255)
    if path is not None:
        plt.savefig(pp_folder(path, _CONST_output))
    if verbose:
        plt.show()

def evaluate_ae(num_samples, weight_path, cropping, env_name):
    
    ae_model = CNNControlAE(lambda u: sparsity_loss(u, 0.05, 0.00), 0.0)
    ae_model.load_state_dict(torch.load(weight_path, map_location='cpu'))

    images, _ = generate_samples(num_samples, environment=env_name)
    ball_bars = []

    for img in images:
        ball_bars.append(img[cropping[0]:cropping[1], cropping[2]:cropping[3]])
    
    # calculate the reconstructed image
    img_recon = list(map(ae_model, ball_bars))
    
    # save images
    base = 'img'
    original = 'orig'
    recon = 'recon'
    if not os.path.exists(base):
        os.mkdir(base)
        if not os.path.exists(os.path.join(base, original)):
            os.mkdir(os.path.join(base, original))
        if not os.path.exists(os.path.join(base, recon)):
            os.mkdir(os.path.join(base, recon))
    
    temp1 = os.path.join(base, original)
    temp2 = os.path.join(base, recon)
    for i in range(len(ball_bars)):
        plt.imsave(fname=os.path.join(temp1, 'img' + str(i)), arr=ball_bars[i], format='png')
        plt.imsave(fname=os.path.join(temp2, 'img' + str(i)), arr=img_recon[i], format='png')

    print('Images were created and saved')
    


