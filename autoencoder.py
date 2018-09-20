import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad 
import torch.utils.data as DT
from losses import ImportanceWeightedBernoulliKLdivLoss, batch_importance


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

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model

def numpy2torch(array):
    return torch.Tensor(array).unsqueeze(0)

def numpy2torch_list(np_list):
    return list(map(numpy2torch, np_list))


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
        self.criterion = ImportanceWeightedBernoulliKLdivLoss(batch_importance)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def fit(self, X, callback=None):
        '''
        X - a trainloader for getting the input frames batch e.g: (N, 84*84*4)
        '''
        for epoch in range(self.epochs):
            for i, data in enumerate(X, 0):
                x = data[0]
                y_real = x.to(self.device) # input is the same as the output (autoencoder)
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
        save_model(self.model, path)


class CNNControlAE(nn.Module):
    '''
    This encoder is based on a CNN and a KL divergence 
    which ensures sparse encoding.
    In case of a new Sparse encoder, just change this one
    and use a clear commit or tag in git.
    Save the configuration in each run.
    '''
    def __init__(self, reg_loss, sigma=1e-6):
        super(CNNControlAE, self).__init__()
        
        self.calculate_reg_loss = reg_loss
        self.sigma = sigma
        latent_size = 50

        # encoder part
        self.conv1 = nn.Conv2d(4, 32, (8, 8), stride=4)
        self.conv2 = nn.Conv2d(32, 64, (4, 4), stride=2)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), stride=1)
        self.conv4 = nn.Conv2d(64, 16, (3, 3), stride=1)
        self.fc_e = nn.Linear(640, latent_size, bias=True)
        
        self.noise = torch.normal(torch.zeros(latent_size), self.sigma * torch.ones(latent_size))

        self.fc_d = nn.Linear(latent_size, 640, bias=True)
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
        self.reg_loss = self.calculate_reg_loss(self.u)
        
        # decoding
        y_ = F.relu(self.fc_d(self.u))
        y_ = y_.view(-1, 16, 8, 5)
        y_ = F.relu(self.deconv1(y_))
        y_ = F.relu(self.deconv2(y_))
        y_ = F.relu(self.deconv3(y_))
        y_ = F.relu(self.deconv4(y_))
        y_ = self.deconv5(y_)
        return torch.sigmoid(y_)
    
    def calculate_feature(self, x):
        self(x) # we do not need the output just the feature at the middle
        return self.u
