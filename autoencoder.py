import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad 
import torch.utils.data as DT


def get_device(cpu_anyway=False, gpu_id=0):
    if cpu_anyway:
        device = torch.device('cpu')
    else:
        device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    return device


def calculate_jacobi_term(x, y_enc, features):
    jacobi_term = 0.0
    for b in range(y_enc.size(0)):
        for i in range(features):
            gradients = grad(y_enc[b, i], x, retain_graph=True, create_graph=True)[0]
            jacobi_term += gradients.pow(2).sum()
            print("In Jacobi: [%d]\r"%i, end='')
    print('\n')
    return jacobi_term


class TrainLoader:

    def __init__(self, images, batch_size):
        self.images = images
        self.batch_size = batch_size
    
    def get_trainloader(self):
        tensors = torch.Tensor(self.images)
        dataset = DT.TensorDataset(tensors)
        return DT.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


class Train:

    def __init__(self, model, device, epochs, lr):
        self.model = model
        self.model.to(device)
        self.device = device
        self.epochs = epochs
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def fit(self, X, callback=None):
        '''
        X - a trainloader for getting the input frames batch e.g: (N, 84*84*4)
        '''
        for epoch in range(self.epochs):
            for i, data in enumerate(X, 0):
                x = data[0]
                x.requires_grad=True
                x = x.to(self.device) # input is the same as the output (autoencoder)
                y_real = x
                y_model = self.model(x)
                loss = self.criterion(y_real, y_model) + self.model.jac_reg
                loss.backward()
                self.optimizer.step()

                print("Iteration: [%d]\r"%i,end='')

                # administration for showing loss and so on
                if callback is not None:
                    callback(epoch, i, loss.item(), x.size(0))


class BasicCae(nn.Module):
 
    def __init__(self, input_size=84*84*4, feature_size=1500):
        super(BasicCae, self).__init__()

        self.input_size = input_size
        self.feature_size = feature_size
        self.jac_reg = 0.0
        self.lin_enc = nn.Linear(self.input_size, self.feature_size)
        self.lin_dec = nn.Linear(self.feature_size, self.input_size)
    
    def forward(self, x): # should be x.requires_grad=True
        self.y_enc = nn.Sigmoid()(self.lin_enc(x))
        self.jac_reg = calculate_jacobi_term(x, self.y_enc, self.feature_size)
        y_out = nn.Sigmoid()(self.lin_dec(self.y_enc))
        return y_out
    
    def calculate_feature(self, x):
        self(x) # we do not need the output just the feature at the middle
        return self.jac_reg


class CNNCae:
    pass

class EdgedetectorAE:
    pass