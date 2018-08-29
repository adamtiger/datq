import gym
import numpy as np
from skimage.color import rgb2grey
from skimage.transform import resize
from skimage.util import crop
from autoencoder import TrainLoader, BasicCae, get_device, Train
import csv
import pandas as pd
from matplotlib import pyplot as plt


def random_image(n, size=(210, 160, 3)):
    '''
    Only for test purposes.
    '''
    images = []
    dones = []
    for _ in range(n):
        img = np.random.randint(0, 256, size)
        images.append(img)
        dones.append(False)
    return images, dones


def preprocess(image, threshold):
    '''
    Preprocesses an image from Atari.
    Grayscale, crop, resize, (normalize)
    '''
    # grayscale
    temp_ = rgb2grey(image)
    # crop
    temp_ = crop(temp_, ((20, 10), (0, 0))) # empirical investigation (crops the playing area)
    # rescale
    temp_ = resize(temp_, (84, 84))
    # binary
    temp_ = abs((temp_ > threshold).astype(np.float32)-1e-6)
    return temp_


def preprocess_batch(images, threshold=0):
    return list(map(lambda img: preprocess(img, threshold), images))


def concatenate(images, dones, length=4):
    '''
    Concatenates length number of consequtive images.
    images - a list of numpy images
    dones - if an episode ends then the next image does not correspond to the current cube
    '''
    rows, cols = images[0].shape[0], images[0].shape[1]
    conc_images = []
    valid = True
    cube = np.zeros((length, rows, cols)) # channel first format in pytorch
    for i, img in enumerate(images):
        if (i+1) % length == 0:
            if valid:
                conc_images.append(cube)
            cube = np.zeros((length, rows, cols))
            valid = True
        cube[i % length, :, :] = img
        valid = valid and not dones[i]
    return conc_images


def flatten(images):
    return list(map(np.ndarray.flatten, images))


def generate_batch(batch_size, exploration_policy=None, environment='Breakout-v0'):
    '''
    batch_size - the number of images to generate during a run
    exploration_policy - the policy how to explore the environments
    '''
    # create the environment and reset
    env = gym.make(environment)
    state = env.reset()
    
    # use random exploration if nothing is provided
    if exploration_policy is None:
        exploration_policy = (lambda x: np.random.randint(0, env.action_space.n))

    # generate images
    images = []
    dones = []
    for _ in range(batch_size):
        state, _, done, _ = env.step(exploration_policy(state))
        images.append(state)
        dones.append(done)
        if done:
            state = env.reset()
    
    return images, dones


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
        images = preprocess_batch(images)
        images = concatenate(images, dones)
        if flat:
            images = flatten(images)
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

import torch
def plot_input_output(ae_model, path):
    
    ae_model.load_state_dict(torch.load(path))
    imgs, dones = generate_batch(4)
    imgs = preprocess_batch(imgs)
    imgs = concatenate(imgs, dones)
    img = torch.tensor(imgs[0], dtype=torch.float32).view(1, 4, 84, 84)

    y = ae_model(img)
    
    plt.figure(8)
    plt.imshow(img.detach().numpy()[0, 0]*255)
    plt.show()
    
    plt.figure(9)
    plt.imshow(y.detach().numpy()[0, 0]*255)
    plt.show()

