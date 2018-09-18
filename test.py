from ae_train import generate_samples
import numpy as np
from matplotlib import pyplot as plt
import losses

# This module is for debugging some parts of the DACQ architecture

def debug_ImportanceKLdivLoss():
    '''
    Shows a map about the KLdivLoss on the image.
    To see if it is correct or not.
    May be this can help to find problems why it does not work.
    '''
    # generate data
    samples = generate_samples(64, environment='Pong-v0') 
    recon_noise = np.random.rand(64, 4, 108, 84) 
    recon_black = np.zeros((64, 4, 108, 84))
    print('Data was generated.')

    # importance map
    I = samples - np.mean(samples, axis=0)
    I_map = I[0, 0]
    print('I map was generated.')

    # KL map
    eps = 1e-15
    KL_noise = (samples * np.log(samples/(recon_noise + eps) + eps) 
            + (1-samples) * np.log((1 - samples)/(1-recon_noise + eps) + eps))
    KL_black = (samples * np.log(samples/(recon_black + eps) + eps) 
            + (1-samples) * np.log((1 - samples)/(1-recon_black + eps) + eps))
    
    KL_map_noise = KL_noise[0, 0]
    KL_map_black = KL_black[0, 0]
    print('KL maps were generated.')

    original = samples[0][0]

    plt.imshow(original)
    plt.show()

    plt.imshow(I_map * 255.0)
    plt.show()

    plt.imshow(KL_map_noise * 255.0)
    plt.show()

    plt.imshow(KL_map_black * 255.0)
    plt.show()

    plt.imshow(I_map * KL_map_black * 255.0)
    plt.show()

    print('Finished.')

debug_ImportanceKLdivLoss()
