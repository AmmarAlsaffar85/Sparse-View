
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
from model import build_unet
from utils import create_dir, seeding
from sklearn.metrics import mean_absolute_percentage_error
from torchmetrics.regression import MeanAbsolutePercentageError

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
pix1=512
pix2=512
H = 256
W = 256

        
if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("results")

    
    ### This is used to test the network after training not during the training process. During the training 
    ### lets say you have six objects to train the network and one object to test it during the training. My 
    ### proposal will be is to use the profile object for testing during the training process. Now in this file 
    ### I have to provide the result from a seventh object to test the network after the finish of the training 
    ### process. So in this case I will the trained model and I will not perform any training process. To make 
    ### it short I will provide in this file the scatter corrupted projection from the scanner and estimate a 
    ### scatter for it. I have to provide a mask for this, I think I will use the collimator as a mask in this 
    ### file. As a test I will use scatter corrupted projection from either EGSnrc or aRTist. 
    
    fn = '/import/scratch/tmp-ct-3/Ammar/NeuralNetwork/ScatterEstimation_final/Ni/TestBladSvenSmall_200keV/images/images_Mpep.scan'
    ar = array.array('h')
    ar=np.fromfile(open(fn, 'rb'),dtype=np.float32)
    #print('PrimaryIntensity length',len(ap))
    Mpep_Images_Test_aRTist=np.array(ar)
    
    NoImages_Test_aRTist=int(len(Mpep_Images_Test_aRTist)/(pix1*pix2))
    
    
    valid_x = Mpep_Images_Test_aRTist.reshape(NoImages_Test_aRTist,pix1,pix2)
    
    
    ## Reduce the size of your test images 
    
    valid_x_DownSampled=np.empty([NoImages_Test_aRTist,H,W])
   
    
    for i in range (0,NoImages_Test_aRTist):
        valid_x_DownSampled[i,:,:] = skimage.measure.block_reduce(valid_x[i,:,:], (2,2), np.mean)
        
    
    """ Hyperparameters """
    checkpoint_path = "/zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/U-Net/UNet-ScatterEstimation/PytorchVesselConstruction-UNet-DSE-Scatter-Paper-FinalVersion/UNET/files_Ni_2/checkpoint.pth"
    
    #checkpoint_path = "/zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/U-Net/UNet-ScatterEstimation/PytorchVesselConstruction-UNet-DSE-Scatter-Paper-FinalVersion/UNET/files_Ni/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    
    pred_y_all = np.array([])
    
    for i in range (0,NoImages_Test_aRTist):
        
        """ Reading image """
        image=valid_x_DownSampled[i,:,:]                  ## (512, 512)
        x = np.expand_dims(image, axis=0)     ## ( 1, 512, 512)
        x = np.expand_dims(x, axis=0)         ## (1, 1, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device, dtype=torch.float32)
        #print(i,x.size())
        
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
           
            
            
     
            pred_y = np.array(pred_y)
            
            #mape(y, y_pred)
            '''
            difference=abs(pred_y-y)
            print(bb)
            
            plt.figure(1)
            plt.imshow(pred_y)
            
            plt.figure(2)
            plt.imshow(y)
            
            plt.figure(3)
            plt.imshow(abs(pred_y-y))
            
            
            plt.figure(4)
            t1=np.arange(0,256)
            a,=plt.plot(t1,pred_y[int(128),:],"b",linewidth=1.0)
            b,=plt.plot(t1,y[int(128),:],"k--",linewidth=1.0)
            c,=plt.plot(t1,difference[int(128),:],"r",linewidth=1.0)
            plt.legend([a, b, c], ['pred_y', 'y', 'difference' ],prop={'size': 12},loc='lower right')
            plt.grid()
            plt.xlabel ('Pixels')
            plt.ylabel ('µ[m⁻¹]xlength[m]')
            #plt.savefig("/lhome/alsaffar/ctutil/Profile",dpi=600,bbox_inches='tight')
            plt.show()
            '''
        pred_y_1darray=pred_y.reshape(H*W)
        pred_y_all=np.concatenate([pred_y_all, pred_y_1darray])

    """ Saving pred_y_all """
    output_file=open("/import/scratch/tmp-ct-3/Ammar/pred_y_BladeSvenSmall_200keV_WithNormalBladeIncluded.scan","wb")
    float_array=np.float32(pred_y_all)
    float_array.tofile(output_file)
    output_file.close()

    '''
    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")
    '''
    fps = 1/np.mean(time_taken)
    print("FPS: ", fps)
