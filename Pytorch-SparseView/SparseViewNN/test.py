#python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-Swin/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/test.py --model Swin2SR_2block
#python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-Swin/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/test.py --model UNet

import argparse
import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
import array
import numpy as np
import skimage.measure
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import matplotlib.pyplot as plt
from torch.nn import Module
from models.load_models import load_model
from utils import create_dir, seeding
from sklearn.metrics import mean_absolute_percentage_error
from torchmetrics.regression import MeanAbsolutePercentageError
import h5py
#from models.Swin2SR import Swin2SR
#from models.UNet import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                        help='Name of the model.')
      
args = parser.parse_args()   
                
'''
def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    #y_true = y_true > 0.5
    #y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    #y_pred = y_pred > 0.5
    #y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask
'''

def MSE(img1, img2):
        squared_diff = (img1 -img2) ** 2
        summed = np.sum(squared_diff)
        num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
        err = summed / num_pix
        return err
        
if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("results")

    
    """ Load dataset """
    ### Load the training data ####
   
    fn0 = '/lhome/alsaffar/NewProjects/SparseViewInterpolation/training.hdf5'
    AllData_input=h5py.File(fn0,'r')['training'][:,:,:]

    fn1 = '/lhome/alsaffar/NewProjects/SparseViewInterpolation/training.hdf5'
    AllData_label=h5py.File(fn1,'r')['label'][:,:,:]
 
    print('All Data input is =',AllData_input.shape[0])
    print('All Data label is =',AllData_label.shape[0])
   
    ################################
    
    testing=AllData_input[0:10000,:,:]
    testing_label=AllData_label[0:10000,:,:]
    
    """ Hyperparameters """
    
    
    checkpoint_path = "/zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-Swin/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/files/checkpoint_"+args.model+".pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #model = build_unet()
    #model = model.to(device)
    model: Module = load_model(args.model).to(device)
    
    #model_Swin2SR=Swin2SR(img_size=64, embed_dim=64, depths=(4,), num_heads=(4,), window_size=8,mlp_ratio=2.)
    #model_path_Swin2SR = '/zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-Swin/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/files/checkpoint.pth'
    
    model_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict=model_dict, strict=True)
    #model.to(device)
    model.eval()
    #pred_image_swin = model_Swin2SR(image)
    '''
    model_UNet = UNet()
    model_path_UNet =  '/zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod/Pytorch-UNet-SparseViewInterpolation/UNET/files/checkpoint.pth'
     
    model_dict = torch.load(model_path_UNet, map_location=device)
    model_UNet.load_state_dict(state_dict=model_dict, strict=True)
    model_UNet.to(device)
    model_UNet.eval() # without this you get error see https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
    pred_image_UNet = model_UNet(image)
    '''
    #model.load_state_dict(torch.load(checkpoint_path, map_location=device))  ## model.load_state_dict(torch.load(PATH)) this will load the trained model according to
    #model.eval()                                                             ## https://pytorch.org/tutorials/beginner/saving_loading_models.html

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    
    pred_y_all = np.array([])
    
    for i in range (0,testing.shape[0]):
        
        """ Reading image """
        image=testing[i,:,:]                  ## (512, 512)
        x = np.expand_dims(image, axis=0)     ## ( 1, 512, 512)
        x = np.expand_dims(x, axis=0)         ## (1, 1, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device, dtype=torch.float32)
        #print(i,x.size())
        """ Reading mask """
        mask = testing_label[i,:,:]                 ## (512, 512)
        y = np.expand_dims(mask, axis=0)      ## (1, 512, 512)
        y = np.expand_dims(y, axis=0)      ## (1, 1, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device, dtype=torch.float32)

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            #print(x.size())
            pred_y = model(x)
            #pred_y = torch.sigmoid(pred_y) # this has to be reomved for scatter estimation DSE
            total_time = time.time() - start_time
            time_taken.append(total_time)


            #score = calculate_metrics(y, pred_y)
            #metrics_score = list(map(add, metrics_score, score))
            #pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = pred_y.to("cpu")
            pred_y = np.squeeze(pred_y, axis=0)     ## (1,512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
           
            
            
            y=y.to("cpu")
            y = np.squeeze(y, axis=0)     ## (1,512, 512)
            y = np.squeeze(y, axis=0)     ## (512, 512)
            
            mean_abs_percentage_error = MeanAbsolutePercentageError()
            print('mape is =',mean_abs_percentage_error(pred_y, y))
            
            #print(y.size())
            #y_DownSampled = skimage.measure.block_reduce(y[:,:], (2,2), np.mean)
    
            pred_y = np.array(pred_y)
            y = np.array(y)
            bb=MSE(pred_y,y)
            #mape = mean_absolute_percentage_error(pred_y,y)
            
            
            #mape(y, y_pred)
            '''
            difference=abs(pred_y-y)
            print(bb)
            '''
            plt.figure(1)
            plt.imshow(pred_y)
            
            plt.figure(2)
            plt.imshow(image)
            
            plt.figure(3)
            plt.imshow(y)
            
            plt.figure(4)
            plt.imshow(abs(pred_y-y))
            
            
            plt.figure(5)
            t1=np.arange(0,64)
            a,=plt.plot(t1,pred_y[int(32),:],"b",linewidth=1.0)
            b,=plt.plot(t1,y[int(32),:],"k--",linewidth=1.0)
            c,=plt.plot(t1,pred_y[int(32),:]-y[int(32),:],"r",linewidth=1.0)
            plt.legend([a, b, c], ['pred_y', 'y', 'difference' ],prop={'size': 12},loc='lower right')
            plt.grid()
            plt.xlabel ('Pixels')
            plt.ylabel ('µ[m⁻¹]xlength[m]')
            #plt.savefig("/lhome/alsaffar/ctutil/Profile",dpi=600,bbox_inches='tight')
            plt.show()
           
        #pred_y_1darray=pred_y.reshape(H*W)
        #pred_y_all=np.concatenate([pred_y_all, pred_y_1darray])

    
    fps = 1/np.mean(time_taken)
    print("FPS: ", fps)
