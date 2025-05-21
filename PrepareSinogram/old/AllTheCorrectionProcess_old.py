#python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-Swin/PrepareSinogram/AllTheCorrectionProcess.py --projectionfile /lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/projdownatten_R002505.hdf5  --downfactor 20 --savedfile /lhome/alsaffar/NewProjects/SparseViewInterpolation/  --ylimt1 0 --ylimt2 292 --xlimt1 0 --xlimt2 800  --patchsize 64 --correcbatch 256 --model UNet

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

from utils import create_dir, seeding
#from sklearn.metrics import mean_absolute_percentage_error
from torchmetrics.regression import MeanAbsolutePercentageError
from scipy.interpolate import RectBivariateSpline

start_time = time.time()



parser = argparse.ArgumentParser()
parser.add_argument('--projectionfile', type=str, required=True,
                       help='scanner projection file.')
parser.add_argument('--downfactor', type=int, required=True,
                       help='Down factor for sparse view.')
parser.add_argument('--savedfile', type=str, required=True,
                       help='where to save the hdf5 result file')
parser.add_argument('--ylimt1', type=int, required=True,
                       help='y limit to crop proj image.')
parser.add_argument('--ylimt2', type=int, required=True,
                       help='y limit to crop proj image.')  
parser.add_argument('--xlimt1', type=int, required=True,
                       help='x limit to crop proj image.')
parser.add_argument('--xlimt2', type=int, required=True,
                       help='x limit to crop proj image.')  
parser.add_argument('--patchsize', type=int, required=True,
                       help='patch size in which the sinogram image is divided.') 
parser.add_argument('--correcbatch', type=int, required=True,
                       help='no of images we throw to the trained network to correct.')                                                 
parser.add_argument('--model', type=str, required=True,
                        help='Name of the model.')
                                                                                       
args = parser.parse_args()

Hdf5File=h5py.File(args.projectionfile,'r')
RealScan=Hdf5File["Image"][:,:,:]
noAngles=(RealScan.shape[0])-1
print(RealScan.shape)
print('noAngles=',noAngles)
pixres1=RealScan.shape[1]
pixres2=RealScan.shape[2]

SourceAxis=Hdf5File["DistanceSourceAxis"][:]
SourceDetector=Hdf5File["DistanceSourceDetector"][:]
PixelSizeX=Hdf5File["DetectorPixelSizeX"][:]
PixelSizeY=Hdf5File["DetectorPixelSizeY"][:]
angles=Hdf5File["Angle"][:]

Hdf5File.close()



#### Normalize the projections data ####

maxelement=numpy.max(RealScan[:,:,:])

RealScan=numpy.divide(RealScan,maxelement)

print('the max value in real scan is =',numpy.max(RealScan[:,:,:]))



#### Create a sinogram from projection data ####
originalsinogram=numpy.zeros((pixres1,int(noAngles),pixres2))

for i in range (0,pixres1):
    for j in range (0,noAngles):
         originalsinogram[i,j,:] = RealScan[j,i,:]

   
plt.figure(1)
plt.title('originalsinogram')
plt.imshow(originalsinogram[0,:,:])

 

#### Generate sparse view sinogram #####

sinogram=numpy.empty((pixres1,int(noAngles/args.downfactor),pixres2))

for i in range(0,pixres1):
    k=0
    for j in range (0,noAngles,args.downfactor):
        #print(j,k)
        sinogram[i,k,:]=originalsinogram[i,j,:]
        k+=1
 

plt.figure(2)
plt.title('sparse view sinogram')
plt.imshow(sinogram[0,:,:])

#### retrieve the projection data from sparse view sinogram ###

projectionsagain_sparseview=numpy.zeros((int(noAngles/args.downfactor),pixres1,pixres2))

for i in range (0,int(noAngles/args.downfactor)):
    for j in range (0,pixres1):
        projectionsagain_sparseview[i,j,:] = sinogram[j,i,:]


  
plt.figure(3)
plt.title('projectionsagain_sparseview')
plt.imshow(projectionsagain_sparseview[0,:,:])
#plt.show()
 
print(projectionsagain_sparseview.shape[0],projectionsagain_sparseview.shape[1],projectionsagain_sparseview.shape[2])




'''
#### Create the hdf5 file for the training and label patches ####

print ("Creating HDF5 file for the Sparse View Projs")
   
file_out_hdf5=h5py.File(args.savedfile+'SparseViewProjs','w')
file_out_hdf5.create_dataset("Type",data=[83,105,109,112,108,101,67,111,110,101,66,101,97,109,67,84,73,
                                          109,97,103,101,83,101,113,117,101,110,99,101], shape=(29,1))
file_out_hdf5.create_dataset("Dimension", data=[65,116,116,101,110,117,97,116,105,111,110], shape = (11,1))
file_out_hdf5.create_dataset("DetectorPixelSizeX", data=pixres1, shape = (1,))
file_out_hdf5.create_dataset("DetectorPixelSizeY", data=pixres2, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceAxis", data=SourceAxis, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceDetector", data=SourceDetector, shape = (1,))
file_out_hdf5.create_dataset("Angle", data=angles)
images_out = file_out_hdf5.create_dataset("Image",shape=(projectionsagain_sparseview.shape[0],
                                                         projectionsagain_sparseview.shape[1],
                                                         projectionsagain_sparseview.shape[2]), 
                                                         dtype='float32', data=projectionsagain_sparseview[:,:,:])
                                                         
file_out_hdf5.close()
'''
 

'''
sinogram=numpy.zeros((pixres1,int(noAngles/args.downfactor),pixres2))
for i in range (0,pixres1):
    k=0
    for j in range (0,args.downfactor,noAngles):
         sinogram[i,k,:] = RealScan[j,i,:]
         k+=1

       
plt.figure(4)
plt.title('sparse view')
plt.imshow(sinogram[0,:,:])
'''    
  

    


#### Linearly interpolate the sparse view sinogram to get it in the same size as the original sinogram ####

### Pix1 is 576 and Pix2 is 800 , pixupper1=2304 , pixupper2=3200
pix1=sinogram.shape[1]
pix2=sinogram.shape[2] 
pixupper1=noAngles
pixupper2=pixres2


print(pix1,pix2,pixupper1,pixupper2)

#all_data2=numpy.array([]) # this will create numpy array
all_data2=[] # this will create list


arguments=[(i,sinogram[i,:,:],pix1,pix2,pixupper1,pixupper2) for i in range(sinogram.shape[0])] 
with Pool(30) as p:              
    all_data2=p.starmap(arrang_array,arguments)

all_data2.sort(key=takeSecond)

#############################################
#for i in range (len(all_data2)):
    #print('Display Keys for Smoothing=',all_data2[i][1])
### Convert List of Smoothing to array ######

res = numpy.array(all_data2,dtype=object) # this will convert list to numpy.array

#############################################

#print('all_data2=',res[:,0])

### Remove the Keys after ordering it #######

res_final=res[:,0]   # the [:,0] represents the slices while [:,1] represents the keys

res_final=abs(res_final)
#res_f = np.array(res_final)


sinograminterpolated_ = numpy.empty((sinogram.shape[0],pixupper2,pixupper1))

#res_final = res_final. astype('float32')

for i in range (0,sinogram.shape[0]):
     #final[i,:,:]= numpy.float32(res_final[i])
      sinograminterpolated_[i,:,:]= res_final[i]


sinograminterpolated=sinograminterpolated_.transpose(0,2,1)

plt.figure(5)
plt.title('Interpolated sinogram')
plt.imshow(sinograminterpolated[0,:,:])
#plt.figure(2)
#plt.title('the difference between original sinogram and interpolated one')
#plt.imshow(abs(originalsinogram[0,:,:]-sinograminterpolated[0,:,:]))
plt.show()


#### Extract patches from the interpolated sinogram ####

# Define patch size and stride
patch_size = (args.patchsize, args.patchsize)
stride = 10
patches_all_interp_siogram=numpy.zeros((10000000,args.patchsize,args.patchsize))


for i in range (0,sinograminterpolated[:,:,:].shape[0]):
    
    image=sinograminterpolated[i,:,:]
    
    # Extract patches using view_as_windows
    
    patches = view_as_windows(image, patch_size, step=stride)
    original_shape = patches.shape 
    #print('original_shape=',original_shape)
    # Reshape patches to get a list of 50x50 patches
    
    patches = patches.reshape(-1, *patch_size)
    
    
    
    # Print the number of patches extracted
    #print(f"Extracted {len(patches)} patches of size {patch_size} with stride {stride}.")

    # Example: Access the first patch
    #first_patch = patches[0]
    #print("First patch shape:", first_patch.shape)  
   
    #plt.figure(2)
    #plt.imshow(patches[3500])

    
    patches_all_interp_siogram[patches.shape[0]*i:patches.shape[0]*(i+1),:,:]=patches
    
    #print('list small=',patches.shape[0]*i,patches.shape[0]*(i+1))
 
    limit=patches.shape[0]*(i+1)
    
patches_all_interp_siogram=patches_all_interp_siogram[0:limit]

print('The size of the patches from interpolated sinogram=',patches_all_interp_siogram.shape)

### Correct the patches (throw everything to the NN) ###

""" Load the checkpoint """
checkpoint_path = "/zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-Swin/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/files/checkpoint_"+args.model+".pth"

#checkpoint_path = "/zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-Swin/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/files/checkpoint_UNet_DownFactor10.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup model, optimizer and loss function.
model: Module = load_model(args.model).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)  ## model.load_state_dict(torch.load(PATH)) this will load the trained model according to
model.eval()                                                             ## https://pytorch.org/tutorials/beginner/saving_loading_models.html
time_taken = []

   
pred_y_all = numpy.empty((patches_all_interp_siogram.shape[0],patches_all_interp_siogram.shape[1],patches_all_interp_siogram.shape[2])) 

print('patches size to check is =',patches_all_interp_siogram.shape[0],patches_all_interp_siogram.shape[1],patches_all_interp_siogram.shape[2])


image = numpy.expand_dims(patches_all_interp_siogram, axis=1)
#image = numpy.expand_dims(image, axis=1)

print('image size is =',image.shape)
image = image.astype(numpy.float32)
xx = torch.from_numpy(image)
#print('xx size is =',xx.size(dim=0))

start_time = time.time()

for i in range (0,int(xx.size(dim=0)/args.correcbatch)): 
        """ Reading image """
        x=xx[i*args.correcbatch:(i+1)*args.correcbatch,:,:,:]                  ## (512, 512)
        x = x.to(device, dtype=torch.float32)
        
        with torch.no_grad():
            """ Prediction and Calculating FPS """
            
            pred_y = model(x)
            pred_y = pred_y.to("cpu")
            pred_y = numpy.squeeze(pred_y, axis=(1))     ## (1,512, 512)
            
       
        pred_y_all[i*args.correcbatch:(i+1)*args.correcbatch,:,:]=pred_y
        
        #time_taken.append(total_time)
        
total_time = time.time() - start_time
print('done correcting the patches')
print(total_time)

#### convert the correct patches to the correct sinogram image #####

reconstructed_image_all = numpy.zeros_like(sinograminterpolated[:,:,:])

eachpatchsize=int(patches_all_interp_siogram[:,:,:].shape[0]/sinograminterpolated[:,:,:].shape[0])

print ('eachpatchsize=',eachpatchsize)

#original_shape= (146, 33, 50, 50)

for k in range (0,sinograminterpolated[:,:,:].shape[0]):

    patches=pred_y_all[k*eachpatchsize:(k+1)*eachpatchsize,:,:]

    patches_ = patches.reshape(original_shape)

    # Get the shape of the patches array
    num_patches_y, num_patches_x, _, _ = patches_.shape

    # Initialize an empty array to reconstruct the image
    reconstructed_image = numpy.zeros_like(sinograminterpolated[k,:,:])
    overlap_count = numpy.zeros_like(sinograminterpolated[k,:,:])  # To count how many patches contribute to each pixel

    # Reconstruct the image by placing patches in their correct positions
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate the position of the patch in the original image
            y_start = i * stride
            y_end = y_start + patch_size[0]
            x_start = j * stride
            x_end = x_start + patch_size[1]
 
            # Add the patch to the reconstructed image
            reconstructed_image[y_start:y_end, x_start:x_end] += patches_[i, j]
            overlap_count[y_start:y_end, x_start:x_end] += 1

    # Average the overlapping regions
    reconstructed_image /= overlap_count
    
    # Verify if the reconstruction matches the original image
    print("Reconstruction matches original image:", numpy.allclose(sinograminterpolated[k,:,:], reconstructed_image))
    reconstructed_image_all[k,:,:]=reconstructed_image
    
    
plt.figure(1)






#### retrieve the projection data from corrected sinogram ###

projectionsagain_corrected=numpy.zeros((noAngles,pixres1,pixres2))

for i in range (0,noAngles):
    for j in range (0,pixres1):
        projectionsagain_corrected[i,j,:] = reconstructed_image_all[j,i,:]



plt.figure(3)
plt.title('projectionsagain_corrected')
plt.imshow(projectionsagain_corrected[0,:,:])
plt.show()


'''
#### Create a sinogram from projection data ####
originalsinogram=numpy.zeros((pixres1,int(noAngles),pixres2))

for i in range (0,pixres1):
    for j in range (0,noAngles):
         originalsinogram[i,j,:] = RealScan[j,i,:]

'''
plt.figure(1)
plt.title('Original Sinogram')
plt.imshow(originalsinogram[0,:,:])
plt.figure(2)
plt.title('Interpolated Sinogram')
plt.imshow(sinograminterpolated[0,:,:])
plt.figure(3)
plt.title('The Difference Between Original Sinogram and Interpolated One')
plt.imshow(abs(originalsinogram[0,:,:]-sinograminterpolated[0,:,:]))

plt.figure(4)
plt.title('Difference between corrected siogram and interpolated one')
plt.imshow(reconstructed_image_all[0,:,:]-sinograminterpolated[0,:,:])
plt.show()


#### Create the hdf5 file for the training and label patches ####

print ("Creating HDF5 file from training data")
   
file_out_hdf5=h5py.File(args.savedfile+'CorrectedProjections.hdf5','w')
file_out_hdf5.create_dataset("Type",data=[83,105,109,112,108,101,67,111,110,101,66,101,97,109,67,84,73,
                                          109,97,103,101,83,101,113,117,101,110,99,101], shape=(29,1))
file_out_hdf5.create_dataset("Dimension", data=[65,116,116,101,110,117,97,116,105,111,110], shape = (11,1))
file_out_hdf5.create_dataset("DetectorPixelSizeX", data=PixelSizeX, shape = (1,))
file_out_hdf5.create_dataset("DetectorPixelSizeY", data=PixelSizeY, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceAxis", data=SourceAxis, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceDetector", data=SourceDetector, shape = (1,))
file_out_hdf5.create_dataset("Angle", data=angles)
images_out = file_out_hdf5.create_dataset("Image",shape=(projectionsagain_corrected[:,:,:].shape[0],
                                                         projectionsagain_corrected[:,:,:].shape[1],
                                                         projectionsagain_corrected[:,:,:].shape[2]), 
                                                         dtype='float32', data=projectionsagain_corrected[:,:,:])
                                                         

file_out_hdf5.close()

exit()






