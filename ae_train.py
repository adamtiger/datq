import gym
import numpy as np
from skimage.color import rgb2grey
from skimage.transform import resize
from skimage.util import crop
from autoencoder import TrainLoader, BasicCae, get_device, Train


def random_image(n, size=(210, 160, 3)):
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
    temp_ = crop(temp_, ((20, 0), (0, 0)))
    # rescale
    temp_ = resize(temp_, (84, 84))
    # binary
    temp_ = (temp_ > threshold).astype(np.float32)
    return temp_


def preprocess_batch(images, threshold=20):
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
    cube = np.zeros((rows, cols, length))
    for i, img in enumerate(images):
        if (i+1) % length == 0:
            if valid:
                conc_images.append(cube)
            cube = np.zeros((rows, cols, length))
            valid = True
        cube[:, :, i % 4] = img
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
    state, _, done, _ = env.reset()
    
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


def train_bae(lr, iterations, epochs, outer_batch, inner_batch, gpu_id=-1):

    # function for following up the algorithm performance
    history = {'iteration':[], 'loss':[], 'ci':0}
    def followup_performance(epoch, i, loss_item, x_size):
        itr = history['ci']
        if itr % 1 == 0:
            history['loss'].append(loss_item)
            history['iteration'].append(itr)
        history['ci'] += 1

    device = get_device(cpu_anyway=(gpu_id==-1), gpu_id=gpu_id)
    
    for _ in range(iterations):
        # create a batch for the outer loop
        images, dones = random_image(outer_batch) # generate_batch(outer_batch, environment='Breakout-v0')
        images = preprocess_batch(images)
        images = concatenate(images, dones)
        images = flatten(images)
        trainloader = TrainLoader(images, inner_batch).get_trainloader()
        
        bae = BasicCae(feature_size=200)
        trainer = Train(bae, device, epochs, lr)
        trainer.fit(trainloader, callback=followup_performance)
    
    return bae, history