import gym
import numpy as np
from skimage.color import rgb2grey
from skimage.transform import resize
from skimage.util import crop
from autoencoder import BasicCae


def random_image(n, size=(210, 160, 3)):
    images = []
    for _ in range(n):
        img = np.random.randint(0, 256, size)
        images.append(img)
    return images


def preprocess(image, normalize=False):
    '''
    Preprocesses an image from Atari.
    Grayscale, crop, resize, (normalize)
    '''
    # grayscale
    temp_ = rgb2grey(image)
    # crop
    temp_ = crop(temp_, ((20, 0), (0, 0)))
    # rescale
    temp_ = resize(temp_, (84, 84))
    # normalize
    if normalize:
        dev = temp_ - np.sum(temp_)/(np.size(temp_))
        std = np.sqrt(np.sum(dev * dev)/np.size(dev))
        temp_ = dev / std

    return temp_


def preprocess_batch(images, normalize=False):
    return list(map(lambda img: preprocess(img, normalize), images))


def generate_batch(batch_size, exploration_policy=None, environment='Breakout-v0'):
    '''
    batch_size - the number of images to generate during a run
    exploration_policy - the policy how to explore the environments
    '''

    # create the environment and reset
    env = gym.make(environment)
    state, _, done, _ = env.reset()
    
    # use random exploration if nothing is provided
    if exploration_policy is None:
        exploration_policy = (lambda x: np.random.randint(0, env.action_space.n))

    # generate images
    images = []
    for _ in range(batch_size):
        state, _, done, _ = env.step(exploration_policy(state))
        images.append(state)
        if done:
            state = env.reset()
    
    return images


def train_bae(iterations, outer_batch, inner_batch):
    pass
