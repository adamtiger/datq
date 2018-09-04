import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad 
import torch.utils.data as DT


def get_device(cpu_anyway=False, gpu_id=0):
    '''
    Sets a concrete device to use.
    gpu_id - the id is the same as the id in nvidia-smi
    '''
    if cpu_anyway:
        device = torch.device('cpu')
    else:
        device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    cuda.set_device(gpu_id)
    return device


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
        self.criterion = ImportanceWeightedBCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def fit(self, X, callback=None):
        '''
        X - a trainloader for getting the input frames batch e.g: (N, 84*84*4)
        '''
        for epoch in range(self.epochs):
            for i, data in enumerate(X, 0):
                x = data[0]
                y_real = torch.max(torch.min(x, torch.tensor(1.0)), torch.tensor(0.0))
                y_real = y_real.to(self.device) # input is the same as the output (autoencoder)
                y_model = self.model(y_real)
                
                self.optimizer.zero_grad()
                loss_rec = self.criterion(y_model, y_real)
                loss_reg = self.model.reg_loss
                loss =  loss_reg + loss_rec
                loss.backward()
                self.optimizer.step()

                # administration for showing loss and so on
                if callback is not None:
                    callback(epoch, i, loss_rec.item(), loss_reg.item(), x.size(0))
            print("Epoch (done): [%d]\r"%epoch, end='')
        print("\n")
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)


class BasicCae(nn.Module):
    '''
    This is a simple Contractive Autoencoder.
    The imeplementation uses a direct calculation of the 
    Frobenius norm instead of using the autograd functions.
    Unfortunately, to create deeper CAE requires to calculate 
    the gradients manually (or analytically) and this is not
    flexible.
    '''
 
    def __init__(self, input_size=84*84*4, feature_size=1500):
        super(BasicCae, self).__init__()

        self.input_size = input_size
        self.feature_size = feature_size
        self.reg_loss = 0.0
        self.lin_enc = nn.Linear(self.input_size, self.feature_size)
        self.lin_dec = nn.Linear(self.feature_size, self.input_size)
    
    def forward(self, x): # should be x.requires_grad=True
        x.requires_grad=True
        self.y_enc = nn.Sigmoid()(self.lin_enc(x))
        self.reg_loss = self.jacobi_loss_calc()
        y_out = nn.Sigmoid()(self.lin_dec(self.y_enc))
        return y_out

    def jacobi_loss_calc(self):
        y = self.y_enc
        sigmoid_der = y * (1-y)
        w = list(self.lin_enc.parameters())[0]
        sigmoid_der_2 = sigmoid_der**2
        w_2 = w**2
        return sigmoid_der_2.matmul(w_2).sum()
    
    def calculate_feature(self, x):
        self(x) # we do not need the output just the feature at the middle
        return self.y_enc


class CNNSparseAE(nn.Module):
    '''
    This encoder is based on a CNN and a KL divergence 
    which ensures sparse encoding.
    In case of a new Sparse encoder, just change this one
    and use a clear commit or tag in git.
    Save the configuration in each run.
    '''
    def __init__(self, beta, rho):
        super(CNNSparseAE, self).__init__()
        
        self.beta = beta # the factor for the regularization term
        self.rho = rho   # the expected average activation in the encoder layer

        # encoder part
        self.conv1 = nn.Conv2d(4, 32, (8, 8), stride=4)
        self.conv2 = nn.Conv2d(32, 64, (4, 4), stride=2)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), stride=1)
        self.conv4 = nn.Conv2d(64, 16, (3, 3), stride=1)
        self.fc_e = nn.Linear(640, 100, bias=True)

        self.fc_d = nn.Linear(100, 640, bias=True)
        self.deconv1 = nn.ConvTranspose2d(16, 16, (3, 3), stride=1)
        self.deconv2 = nn.ConvTranspose2d(16, 64, (3, 3), stride=1)
        self.deconv3 = nn.ConvTranspose2d(64, 64, (4, 4), stride=2)
        self.deconv4 = nn.ConvTranspose2d(64, 32, (8, 8), stride=4)
        self.deconv5 = nn.ConvTranspose2d(32, 4, (1, 1), stride=1)

        self.u = 0.0
        self.reg_loss = 0.0

    def forward(self, x):
        # encoding
        x_ = F.relu(self.conv1(x))
        x_ = F.relu(self.conv2(x_))
        x_ = F.relu(self.conv3(x_))
        x_ = F.relu(self.conv4(x_))
        x_ = x_.view(x_.size(0), -1) # flatten the 3D tensor to 1D in batch mode
        x_ = F.relu(self.fc_e(x_))
        self.u = x_

        # calculate reg loss
        self.reg_loss = self.u.mean(0).sum() * 0.0 # self.calculate_reg_loss()
        
        # decoding
        y_ = F.relu(self.fc_d(self.u))
        y_ = y_.view(-1, 16, 8, 5)
        y_ = F.relu(self.deconv1(y_))
        y_ = F.relu(self.deconv2(y_))
        y_ = F.relu(self.deconv3(y_))
        y_ = F.relu(self.deconv4(y_))
        y_ = self.deconv5(y_)
        return torch.sigmoid(y_)
        
    def calculate_reg_loss(self):
        rho_ = self.u.mean(0) # average activations for each node
        rho = self.rho
        kl_div = rho * torch.log(rho/rho_) + (1-rho) * torch.log((1-rho)/(1-rho_)) # KL divergence for avg. activation
        return self.beta * kl_div.sum()/rho_.size(0)
    
    def calculate_feature(self, x):
        self(x) # we do not need the output just the feature at the middle
        return self.u


class ImportanceWeightedBCELoss(nn.Module):

    def __init__(self):
        super(ImportanceWeightedBCELoss, self).__init__()
        self.loss = 0.0

    def forward(self, original, reconstructed):
        batch_size = original.size(0)
        I = (original - torch.mean(original))
        BCE = original * torch.log(reconstructed) + (1-original) * torch.log(1-reconstructed)
        self.loss = torch.sum(I * BCE) / batch_size
        return self.loss

