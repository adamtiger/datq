import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class JacobianRegularizer(nn.Module):

    def __init__(self):
        super(JacobianRegularizer, self).__init__()

    def forward(self, input):
        self.loss = 0
        self.value = input
        return input

class PreprocessAtari(nn.Module):

    def __init__(self, crop_h=(10, 210), crop_v=(20, 140)):
        super(PreprocessAtari, self).__init__()
        self.crop_h = crop_h
        self.crop_v = crop_v

    def forward(self, input):
        # steps: gray_scale, stack, crop, scale
        return input


class BasicCae:
 
    def __init__(self, epochs, lr, input_size=86*86*4, feature_size=1500):
        self.model = self.__create_model()
        self.epochs = epochs
        self.input_size = input_size
        self.feature_size = feature_size

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters, lr=lr)
    
    def __create_model(self):
        # layers which will be used later on
        self.prep = PreprocessAtari()
        self.jac_reg = JacobianRegularizer()

        model = nn.Sequential()
        model.add_module('Preprocessing', self.prep)
        model.add_module('Linear_encode', nn.Linear(self.input_size, self.feature_size))
        model.add_module('Sigmoid_encode', nn.Sigmoid())
        model.add_module('JacobyRegularization', self.jac_reg)
        model.add_module('Linear_decode', nn.Linear(self.feature_size, self.input_size))
        model.add_module('Sigmoid_decode', nn.Sigmoid())
        return model
    
    def fit(self, X):
        '''
        X - a trainloader for getting the input frames batch e.g: (N, 4, 84, 84, 4)
        '''
        for epoch in range(self.epochs):
            for i, data in enumerate(X, 0):
                x = data # input is the same as the output (autoencoder)
                y_real = self.prep(x)

                y_model = self.model(x)

                loss = self.criterion(y_real, y_model) + self.jac_reg.loss
                loss.backward()
                self.optimizer.step()

                # administration for showing loss and so on
    
    def calculate_feature(self, x):
        self.model(x) # we do not need the output just the feature at the middle
        return self.jac_reg.value


class CNNCae:
    pass

class EdgedetectorAE:
    pass