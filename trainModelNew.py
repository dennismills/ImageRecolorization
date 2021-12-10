#!/usr/bin/env python
# coding: utf-8

import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import random

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
    We assume that all the images we will be using
    in this project will be loaded in (with glob)
    from a dataset folder that is in the project
    directory. This can be easily changed by editing the
    path variable.
'''
path = "dataset/"
paths = []

os.chdir(path) # Changes into the dataset folder

'''
    On Mac, Python3 creates a folder called .DS_Store.
    This folder won't have any image data, so filtering for it
    is not required, but it throws off the number of subfolders
    we count. It also is just a waste of time to search through later,
    so we exclude it from our paths.
'''
dirs = [dir_ for dir_ in os.listdir() if dir_ != ".DS_Store"] # Gets a list of all the subfolders in the path
'''
    If you want to see how many folders you are sampling from,
    uncomment the line below. Our model will sample
    from EVERY subfolder in the dataset directory,
    which makes it very easy to add images.
'''
# print("Number of folders found {}".format(len(dirs)))


paths = [] # A variable to store the paths we load from
maxFromFolder = 500 # The maximum number of images we will load from each folder

'''
    ******NOTE******
    The program will crash if each folder has less than
    500 images in the folder. If you are reproducing our results,
    then please be sure to adjust this value, or adjust
    the code so that way it can work with the smallest number
    of images in any of the subfolders.
'''

os.chdir("../") # Changes the directory to the root folder
count = 0 # A count variable that is incremented every time a new image is loaded

'''
    The code below will only look for .jpg files
    in the current directory. It could be modified
    to look for all image types, but we were only using
    .jpg images, so we never had a need to make it more flexible than this.
'''
for directory in dirs: # Looks in the current path
    currentPath = os.getcwd() # Gets the current working directory and saves it
    os.chdir(path + directory) # Changes directories into the current path
    sizeOfFolder = len(os.listdir()) # Gets the number of elements in the folder
    os.chdir(currentPath) # Changes into the subpath
    randIndices = random.sample(range(0, sizeOfFolder - 1), maxFromFolder) # Gets a random sample of indices of 500 images to use
    allPaths = glob.glob(path + directory + "/*.jpg") # Gets all the paths from the folder that are jpgs
    for index in randIndices: # For each index we've selected
        paths.append(allPaths[index]) # Append that image path to the paths list
        count += 1 # increment the counter

print("Number of images being loaded: {}".format(len(paths))) # Displays the number of paths

np.random.seed(123) # Uses a seed for predictable results
pathsSubset = np.random.choice(paths, len(paths), replace = False) # choosing 1000 images randomly
randomIndex = np.random.permutation(len(paths))

cutoff = int(0.8 * len(paths)) # By default we will only load in about 80% of the data for the train / test split

trainIndices = randomIndex[:cutoff] # choosing the first 80% as training set
valueIndices = randomIndex[cutoff:] # choosing last 20% as validation set
trainPaths = pathsSubset[trainIndices] # gets the train paths
valuePaths = pathsSubset[valueIndices] # gets the value paths

SCALED_IMAGE_SIZE = 256 # This is the scaled image size that we will be using
# Either the images are downampled into this size, or they are upscaled into this size


'''
    The code below is a class that is designed
    so that we can use it as a Generator for the data
    loading process. It is a generator in the sense that
    it is designed to yield a result each time next() is applied
    to it so that way we can reduce on the spatial complexity overhead
    of the project, not a generator in the GAN sense.
    
    See this link for more information on the generator design pattern:
    https://wiki.python.org/moin/Generators
'''
class DatasetLoader(Dataset):
    def __init__(self, paths, split = 'train'):
        if split == 'train': # If we are training the data
            
            '''
                We are using the Compose object since it can store
                information about transformations that we can apply to each
                image. This is just a hand object that is used as a utility,
                and not for anything else. It can be excluded if need be.
            '''
            self.transforms = transforms.Compose([
                transforms.Resize((SCALED_IMAGE_SIZE, SCALED_IMAGE_SIZE),  Image.BICUBIC), # Scales the image to the desired size
                transforms.RandomHorizontalFlip(), # This can decide to randomly flip the data incoming
                transforms.RandomRotation(360.0) # This line can be used to add a random 360 degree rotation
            ])
            
            '''
                See the link below for a complete list of all the different
                types of data augmentation that are available in the
                transforms object built into pytorch.
                
                https://pytorch.org/vision/stable/transforms.html
            '''
            
        elif split == 'val': # If we are in the validation (testing) phase, we don't want to augment the data.
            '''
                Since we can only compare tensor objects of the same
                dimensionality, we still have to scale the validation set in the same
                way that we did the training set. Otherwise we would get a dimensionality
                mismatch that could later cause issues.
            '''
            self.transforms = transforms.Resize((SCALED_IMAGE_SIZE, SCALED_IMAGE_SIZE),  Image.BICUBIC)
        
        self.split = split # Stores the split
        self.size = SCALED_IMAGE_SIZE # Stores the image sizes
        self.paths = paths # Stores the paths for the images we wish to load (since we are using the generator pattern)
    
    '''
        This function allows us to get the next
        item from the generator
    '''
    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB") # Loads in the image and converts to RGB if it isn't already
        img = self.transforms(img) # Uses the transform object to augment the image
        img = np.array(img) # Converts the image to a numpy array
        LABImage = rgb2lab(img).astype("float32") # Converting RGB to the LAB color space as a 32 bit floating point np array
        LABImage = transforms.ToTensor()(LABImage) # transforms it into a tensor object to work with pytorch
        L = LABImage[[0], ...] / 50. - 1. # Between -1 and 1
        ab = LABImage[[1, 2], ...] / 110. # Between -1 and 1
        
        return {'L': L, 'ab': ab} # Returns a dictionary with the key value pairs to address the L and the AB channels respectively
    
    '''
        This is a function that returns the number of images that are stored in the generator
    '''
    def __len__(self):
        return len(self.paths)


'''
    This function is responsible for making building the dataloader. Since we can load
    a ton of images into the model, we should take advantage of the
    parallel nature of PyTorch and user their parallel data loaders when possible.
'''
def makeDataloaders(batchSize = 16, numberOfWorkers = 4, pinMemory = True, **kwargs):
    dataset = DatasetLoader(**kwargs) # creates the data loader
    dataloader = DataLoader(dataset, batch_size = batchSize, num_workers = numberOfWorkers,
                            pin_memory = pinMemory) # provides the data loader with the dataset and the params
    return dataloader # returns the loader object


'''
    Below we create the training data loader,
    as well as the validation data loader
'''
trainingDataLoader = makeDataloaders(paths = trainPaths, split = 'train')
validationDataLoader = makeDataloaders(paths = valuePaths, split = 'val')

data = next(iter(trainingDataLoader)) # We use the generator pattern to get the training data set
Ls, abs_ = data['L'], data['ab'] # We exctract the L and AB channels from the dataset


class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)
        
        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)
    
    def forward(self, x):
        return self.model(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2) 
                          for i in range(n_down)] # the 'if' statement is taking care of not using
                                                  # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)] # Make sure to not use normalization or
                                                                                             # activation for the last layer of the model
        self.model = nn.Sequential(*model)                                                   
        
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True): # when needing to make some repeatitive blocks of layers,
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]          # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
    
    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)
    
    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss


def init_weights(net, init='norm', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
            
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model


class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, 
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        
        if net_G is None:
            self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
    
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        
    def forward(self):
        self.fake_color = self.net_G(self.L)
    
    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)
    
def visualize(model, data):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()

    import os
    cwd = os.getcwd()
    if not os.path.exists(cwd + "/recolored"):
        os.mkdir(cwd + "/recolored/")
    os.chdir(cwd + "/recolored/")
    fig.savefig("colorization{}.png".format(time.time()))
    os.chdir("../")
        
def log_results(loss_meter_dict, storageDict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")
        if loss_name not in storageDict:
            storageDict[loss_name] = []
        storageDict[loss_name].append(loss_meter.avg)

def saveModel(model, path):
    torch.save(model.state_dict(), path)
    
def loadModel(path):
    model = MainModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

import yagmail
from datetime import datetime

def train_model(model, trainingDataLoader, epochs, stroageDict):
    data = next(iter(validationDataLoader)) # getting a batch for visualizing the model output after fixed intrvals
    epochStart = time.time()
    times = []
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in tqdm(trainingDataLoader):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1

        log_results(loss_meter_dict, stroageDict) # function to print out the losses
        print("------------- SAVING OUTPUT ---------------")
        visualize(model, data) # function displaying the model's outputs
        epochEnd = time.time()
        elapsed = epochEnd - epochStart
        remainingEpochs = epochs - e
        epochStart = epochEnd
        print("Estimated time remaining: {}h".format((elapsed * remainingEpochs) / (60.0 * 60)))

model = MainModel()

storage = {}
totalEpochs = 250
train_model(model, trainingDataLoader, totalEpochs, storage)
output = []
epochRange = [i + 1 for i in range(totalEpochs)]

fig = plt.figure()
import random
for key, value in storage.items():
    color = random.random(), random.random(), random.random()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(epochRange, value, c = color, label = str(key))
    plt.legend(loc = "upper right")

plt.savefig("lossChart.png")
saveModel(model, "decemberModel.pt")




