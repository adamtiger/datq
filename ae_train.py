import gym
import csv
import torch
import numpy as np
import pandas as pd
from skimage.color import rgb2grey
from skimage.transform import resize
from skimage.util import crop
from autoencoder import TrainLoader, BasicCae, get_device, Train
from environment import Environment as E
from matplotlib import pyplot as plt


def generate_batch(batch_size, environment='Breakout-v0'):
    '''
    batch_size - the number of images to generate during a run
    environment - name of the OpenAI like environment
    '''
    # create the environment and reset
    env = gym.make(environment)

    # generate images
    return E.generate_random_trajectory(batch_size, env)


def train_ae(ae_model, folder, lr, iterations, epochs, outer_batch, inner_batch, flat=False, gpu_id=-1, callback=None):
    '''
    The training function for AutoEncoders. This is a general function.
    callback - function for handling the administrative data
    '''
    device = get_device(cpu_anyway=(gpu_id==-1), gpu_id=gpu_id)
    
    for i in range(iterations):
        print("Iteration (starting): [%d]"%i)

        # create a batch for the outer loop
        images, dones = generate_batch(outer_batch, environment='Breakout-v0') # random_image(outer_batch)
        images = E.preprocess_batch(images)
        images = E.concatenate(images, dones)
        if flat:
            images = E.flatten(images)
        trainloader = TrainLoader(images, inner_batch).get_trainloader()
        
        trainer = Train(ae_model, device, epochs, lr)
        trainer.fit(trainloader, callback=callback)
        if (i+1) % 10 == 0:
            trainer.save(folder + "/model_weights" + str(i) + ".pt")
    
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

def plot_learning_curve(file):
    '''
    Plots the total loss at each iteration.
    file - the csv file containing (iteration, loss_rec, loss_reg)
    '''
    df = pd.read_csv(file)
    x = df['iteration'].values
    y_rec = df['loss_rec']
    y_reg = df['loss_reg']
    plt.plot(x, y_rec, 'yo', x, y_reg, 'ro')
    plt.show()

def plot_input_output(ae_model, path=None):
    
    if path is not None:
        ae_model.load_state_dict(torch.load(path, map_location='cpu'))
    imgs, dones = generate_batch(4)
    imgs = E.preprocess_batch(imgs)
    imgs = E.concatenate(imgs, dones)
    img = torch.tensor(imgs[0], dtype=torch.float32).view(1, 4, 108, 84)

    y = ae_model(img)
    
    plt.figure(8)
    plt.imshow(img.detach().numpy()[0, 2]*255)
    plt.show()
    
    plt.figure(9)
    plt.imshow(y.detach().numpy()[0, 2]*255)
    plt.show()

