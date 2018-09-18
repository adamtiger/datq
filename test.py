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
    samples, _ = generate_samples(68, environment='Pong-v0')
    samples = np.array(samples) 
    print(np.max(samples[50, 0]), np.min(samples[50, 0]))
    recon_noise = np.random.rand(64, 4, 108, 84) 
    recon_black = np.zeros((64, 4, 108, 84)) + 0.12
    print('Data was generated.')

    # importance map
    I = np.abs(samples - np.mean(samples, axis=0))
    I_map = I[50, 0]
    print('I map was generated.')

    # KL map
    eps = 1e-15
    KL_noise = (samples * np.log(samples/(recon_noise + eps) + eps) + (1-samples) * np.log((1 - samples)/(1-recon_noise + eps) + eps))
    KL_black = (samples * np.log(samples/(recon_black + eps) + eps) + (1-samples) * np.log((1 - samples)/(1-recon_black + eps) + eps))
    
    KL_map_noise = KL_noise[50, 0]
    KL_map_black = KL_black[50, 0]
    print('KL maps were generated.')

    original = samples[50, 0]
    '''
    plt.imshow(original)
    plt.show()

    plt.imshow(I_map * 255.0)
    plt.show()

    plt.imshow(KL_map_noise * 10.0)
    plt.show()

    plt.imshow(KL_map_black * 10.0)
    plt.show()
    
    plt.imshow(I_map * KL_map_noise * 255.0)
    plt.show()

    plt.imshow(I_map * KL_map_black * 255.0)
    plt.show()
    '''
    loss_noise = np.sum(I*KL_noise)/I.shape[0]
    loss_black = np.sum(I*KL_black)/I.shape[0]
  
    print('Losses: %.3f, %.8f'%(loss_noise, loss_black))

debug_ImportanceKLdivLoss()

def draw_KL():
    eps = 1e-15
    rec = 0.0012
    sam = np.array([t/10.0 for t in range(10)])
    kl = sam * np.log(sam/(rec + eps) + eps) + (1-sam) * np.log((1 - sam)/(1-rec + eps) + eps)

    plt.plot(sam, kl, 'ro')
    plt.show()

#draw_KL()

   