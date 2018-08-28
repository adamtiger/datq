import torch
import torch.cuda as cuda
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
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def fit(self, X, callback=None):
        '''
        X - a trainloader for getting the input frames batch e.g: (N, 84*84*4)
        '''
        for epoch in range(self.epochs):
            for i, data in enumerate(X, 0):
                x = data[0]
                x = x.to(self.device) # input is the same as the output (autoencoder)
                y_real = x
                y_model = self.model(x)

                loss_reconst = self.criterion(y_model, y_real)
                loss_reg = self.model.reg_loss
                loss = loss_reconst + loss_reg
                loss.backward()
                self.optimizer.step()

                # administration for showing loss and so on
                if callback is not None:
                    callback(epoch, i, loss_reconst.item(), loss_reg.item(), x.size(0))
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
    
    def __init__(self, beta, rho):
        super(CNNSparseAE, self).__init__()
        
        self.beta = beta
        self.rho = rho

        # encoder part
        self.conv1 = nn.Conv2d(4, 64, (3, 3), stride=3)
        self.conv2 = nn.Conv2d(64, 128, (4, 4), stride=4)
        self.fc_e = nn.Linear(6272, 2500, bias=True)

        self.fc_d = nn.Linear(2500, 6272, bias=True)
        self.deconv1 = nn.ConvTranspose2d(128, 64, (4, 4), stride=4)
        self.deconv2 = nn.ConvTranspose2d(64, 4, (3, 3), stride=3)

        self.u = 0.0
        self.reg_loss = 0.0

    def forward(self, x):
        # encoding
        x_ = F.relu(self.conv1(x))
        x_ = F.relu(self.conv2(x_))
        x_ = x_.view(x_.size(0), -1) # flatten the 3D tensor to 1D in batch mode
        self.u = F.softmax(self.fc_e(x_))

        # calculate reg loss
        self.reg_loss = self.calculate_reg_loss()
        
        # decoding
        y_ = torch.tanh(self.fc_d(self.u))
        y_ = y_.view(-1, 128, 7, 7)
        y_ = F.relu(self.deconv1(y_))
        return F.relu(self.deconv2(y_))
        
    def calculate_reg_loss(self):
        rho_ = self.u.mean(0) # average activations for each node
        rho = self.rho
        kl_div = rho * torch.log(rho/rho_) + (1-rho) * torch.log((1-rho)/(1-rho_))
        return self.beta * kl_div.sum()
    
    def calculate_feature(self, x):
        self(x) # we do not need the output just the feature at the middle
        return self.u

class EdgedetectorAE:
    pass
