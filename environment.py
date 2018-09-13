import numpy as np
from skimage.color import rgb2grey
from skimage.transform import resize
from skimage.util import crop


class Environment:

    def __init__(self, gym_environment, length=4, threshold=0):
        self.env = gym_environment
        self.length = length
        self.threshold = threshold
        img = Environment.preprocess(self.env.reset(), threshold)
        self.observation_as_list = [img] * length
        self.observation = np.array(self.observation_as_list)
    
    @staticmethod
    def preprocess(image, threshold):
        '''
        Preprocesses an image from Atari.
        Grayscale, crop, resize, (normalize)
        '''
        # grayscale
        temp_ = rgb2grey(image)
        # crop
        temp_ = crop(temp_, ((20, 12), (0, 0))) # empirical investigation (crops the playing area)
        # rescale
        temp_ = resize(temp_, (108, 84))
        # scale between 0 and 1 but not normalized for the whole image instead pixelwise
        temp_ = temp_ / 255.0
        return temp_
    
    @staticmethod
    def preprocess_batch(images, threshold=0):
        return list(map(lambda img: Environment.preprocess(img, threshold), images))
    
    @staticmethod
    def concatenate(images, dones, length=4):
        assert len(images) == len(dones)
        concatenated = []
        episode_end = []
        for idx in range(len(images)-length):
            image_batch = np.array(images[idx:(idx+length)])
            done_batch = (np.array(dones[idx:(idx+length-1)], dtype=np.int32).sum() == 0)
            if done_batch:
                concatenated.append(image_batch)
                episode_end.append(dones[idx+length-1])
        return concatenated, episode_end
       
    @staticmethod
    def flatten(images):
        return list(map(np.ndarray.flatten, images))
    
    @staticmethod
    def generate_random_image(n, size=(210, 160, 3)):
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
    
    @staticmethod
    def generate_random_trajectory(step_number, environment):
        env = environment
        env.reset()

        n = env.action_space.n
        # generate images
        images = []
        dones = []
        for _ in range(step_number):
            state, _, done, _ = env.step(np.random.randint(0, n))
            images.append(state)
            dones.append(done)
            if done:
                env.reset()
    
        return images, dones

    def environment_step(self, policy):
        '''
        policy - function to decide an action, input: the preprocessed images (e.g. 4x84x84)
        '''
        action = policy(self.observation)
        image, reward, done, _ = self.env.step(action)
        fi = Environment.preprocess(image, self.threshold)
        self.observation_as_list.append(fi)
        del self.observation_as_list[0]
        self.observation = np.array(self.observation_as_list)

        if done:
            img = self.env.reset()
            self.observation_as_list = [img] * self.length
        
        return (self.observation, action, reward, done)
    
    def generate_trajectory(self, step_number, policy):
        '''
        Generates a trajectory with a fixed policy.
        '''
        trajectory = []
        for _ in range(step_number):
            item = self.environment_step(policy)
            trajectory.append(item)
        return trajectory