#python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-Swin/PrepareSinogram/AllTheCorrectionProcess.py --projectionfile /lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/projdownatten_R002505.hdf5  --model UNet

def arrang_array(i,data11,pix1,pix2,pixupper1,pixupper2):

  xt=numpy.linspace(0,pix2,pix2)
  yt=numpy.linspace(0,pix1,pix1) 

  X, Y=numpy.meshgrid(xt,yt)

  xt2=numpy.linspace(0,pix2,pixupper2)
  yt2=numpy.linspace(0,pix1,pixupper1)

  #all_data=numpy.zeros(noAngles*pixres*pixres)
  #all_data=numpy.array([])
  #arr_n=numpy.zeros((noAngles,pixupper1,pixupper2))
  #arr=numpy.float32(all_data_return)
  #for k in range (0,noAngles):

  io=numpy.transpose(data11[:,:])
  #print(io.shape)
  ff1=RectBivariateSpline(xt,yt,io)
  arr_n=ff1(xt2,yt2)
  #print('aar_n=',arr_n.shape)
  #data = arr_n.reshape((pixupper1,pixupper2))
  #all_data=numpy.append(all_data,data) 
  #q.put(all_data)
  #res = cv2.resize(io, dsize=(pixupper2, pixupper1), interpolation=cv2.INTER_CUBIC)
  return [arr_n,i]



def takeSecond(elem):
    return elem[1]


# make list multprocess list 
import scipy.interpolate as interp
import h5py
import sys
#import cv2
import h5py 
import numpy as numpy
import matplotlib.pyplot as plt
import multiprocessing 
from multiprocessing import Process, Queue
from scipy.interpolate import interp2d
#from joblib import Parallel, delayed
import time
import array
from multiprocessing import Pool
import argparse
from skimage.util import view_as_windows

import torch
import array
import skimage.measure
#from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import matplotlib.pyplot as plt
from torch.nn import Module
from models.load_models import load_model


#from sklearn.metrics import mean_absolute_percentage_error
from torchmetrics.regression import MeanAbsolutePercentageError
from scipy.interpolate import RectBivariateSpline

print(torch.__version__)
start_time = time.time()



parser = argparse.ArgumentParser()
parser.add_argument('--projectionfile', type=str, required=True,
                       help='scanner projection file.')                                               
parser.add_argument('--model', type=str, required=True,
                        help='Name of the model.')
                                                                                       
args = parser.parse_args()

Hdf5File=h5py.File(args.projectionfile,'r')

patches_training=Hdf5File["training"][:,:,:]
patches_label=Hdf5File["label"][:,:,:]

Hdf5File.close()


print('The size of the patches is=',patches_training.shape)

### Correct the patches (throw everything to the NN) ###

""" Load the checkpoint """
checkpoint_path = "/zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-Swin/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/files/checkpoint_"+args.model+".pth"

#checkpoint_path = "/zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-Swin/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/files/checkpoint_UNet_DownFactor10.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup model, optimizer and loss function.

# Load the state_dict
state_dict = torch.load(checkpoint_path)

# Print the keys in the state_dict
print(state_dict.keys())

# Print the keys expected by the model
#print(model.state_dict().keys())


model: Module = load_model(args.model).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))  ## model.load_state_dict(torch.load(PATH)) this will load the trained model according to
model.eval()                                                             ## https://pytorch.org/tutorials/beginner/saving_loading_models.html
time_taken = []



image = numpy.expand_dims(patches_training, axis=1)
image = numpy.expand_dims(image, axis=1)

print('image size is =',image.shape)
image = image.astype(numpy.float32)
xx = torch.from_numpy(image)
#print('xx size is =',xx.size(dim=0))

start_time = time.time()

for i in range (0,int(xx.size(dim=0))): 
        """ Reading image """
        x=xx[i,:,:,:]                  ## (512, 512)
        x = x.to(device, dtype=torch.float32)
        
        with torch.no_grad():
            """ Prediction and Calculating FPS """
            
            pred_y = model(x)
            pred_y = pred_y.to("cpu")
            pred_y = numpy.squeeze(pred_y, axis=(1))     ## (1,512, 512)
         
        plt.figure(1)
        plt.title('input')
        plt.imshow(patches_training[0,:,:])
          
        plt.figure(2)
        plt.title('pred')
        plt.imshow(pred_y[0,:,:])
       
        plt.figure(3)
        plt.title('pred label')
        plt.imshow(patches_label[0,:,:])
        plt.show()
        





