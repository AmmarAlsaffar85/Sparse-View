#python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/PrepareSinogram/AllTheCorrectionProcess-Fast.py --projectionfile /lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/projdownatten_R002505.hdf5  --downfactor 20 --savedfile /lhome/alsaffar/NewProjects/SparseViewInterpolation/  --ylimt1 0 --ylimt2 292 --xlimt1 0 --xlimt2 800  --patchsize 64 --correcbatch 256 --model UNet

#python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/PrepareSinogram/AllTheCorrectionProcess-Fast.py --projectionfile /lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/projdownatten_R002505.hdf5  --downfactor 20 --savedfile /lhome/alsaffar/NewProjects/SparseViewInterpolation/  --ylimt1 0 --ylimt2 292 --xlimt1 0 --xlimt2 800  --patchsize 64 --correcbatch 256 --model UNet

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
#from models.load_models import load_model
from models.load_models import ModelLoader
from utils import create_dir, seeding
#from sklearn.metrics import mean_absolute_percentage_error
from torchmetrics.regression import MeanAbsolutePercentageError
from scipy.interpolate import RectBivariateSpline
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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


print ("Creating HDF5 file for the Orignal Normalized Projs")
   
file_out_hdf5=h5py.File(args.savedfile+'R002505_OriginalNorm.hdf5','w')
file_out_hdf5.create_dataset("Type",data=[83,105,109,112,108,101,67,111,110,101,66,101,97,109,67,84,73,
                                          109,97,103,101,83,101,113,117,101,110,99,101], shape=(29,1))
file_out_hdf5.create_dataset("Dimension", data=[65,116,116,101,110,117,97,116,105,111,110], shape = (11,1))
file_out_hdf5.create_dataset("DetectorPixelSizeX", data=PixelSizeX, shape = (1,))
file_out_hdf5.create_dataset("DetectorPixelSizeY", data=PixelSizeY, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceAxis", data=SourceAxis, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceDetector", data=SourceDetector, shape = (1,))
file_out_hdf5.create_dataset("Angle", data=angles)
images_out = file_out_hdf5.create_dataset("Image",shape=(RealScan.shape[0],
                                                         RealScan.shape[1],
                                                         RealScan.shape[2]), 
                                                         dtype='float32', data=RealScan[:,:,:])
                                                         
file_out_hdf5.close()

#### Create a sinogram from projection data ####
originalsinogram=numpy.zeros((pixres1,int(noAngles),pixres2))

for i in range (0,pixres1):
    for j in range (0,noAngles):
         originalsinogram[i,j,:] = RealScan[j,i,:]

   
#plt.figure(1)
#plt.title('originalsinogram')
#plt.imshow(originalsinogram[0,:,:])

 

#### Generate sparse view sinogram #####

sinogram=numpy.empty((pixres1,int(noAngles/args.downfactor),pixres2))

for i in range(0,pixres1):
    k=0
    for j in range (0,noAngles,args.downfactor):
        #print(j,k)
        sinogram[i,k,:]=originalsinogram[i,j,:]
        k+=1
 

#plt.figure(2)
#plt.title('sparse view sinogram')
#plt.imshow(sinogram[0,:,:])

#### retrieve the projection data from sparse view sinogram ###

projectionsagain_sparseview=numpy.zeros((int(noAngles/args.downfactor),pixres1,pixres2))

for i in range (0,int(noAngles/args.downfactor)):
    for j in range (0,pixres1):
        projectionsagain_sparseview[i,j,:] = sinogram[j,i,:]


  
#plt.figure(3)
#plt.title('projectionsagain_sparseview')
#plt.imshow(projectionsagain_sparseview[0,:,:])
#plt.show()
 
print(projectionsagain_sparseview.shape[0],projectionsagain_sparseview.shape[1],projectionsagain_sparseview.shape[2])





#### Create the hdf5 file for the training and label patches ####

print ("Creating HDF5 file for the Sparse View Projs")
   
file_out_hdf5=h5py.File(args.savedfile+'SparseViewProjs.hdf5','w')
file_out_hdf5.create_dataset("Type",data=[83,105,109,112,108,101,67,111,110,101,66,101,97,109,67,84,73,
                                          109,97,103,101,83,101,113,117,101,110,99,101], shape=(29,1))
file_out_hdf5.create_dataset("Dimension", data=[65,116,116,101,110,117,97,116,105,111,110], shape = (11,1))
file_out_hdf5.create_dataset("DetectorPixelSizeX", data=PixelSizeX, shape = (1,))
file_out_hdf5.create_dataset("DetectorPixelSizeY", data=PixelSizeY, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceAxis", data=SourceAxis, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceDetector", data=SourceDetector, shape = (1,))
file_out_hdf5.create_dataset("Angle", data=angles[0:noAngles:args.downfactor])
images_out = file_out_hdf5.create_dataset("Image",shape=(projectionsagain_sparseview.shape[0],
                                                         projectionsagain_sparseview.shape[1],
                                                         projectionsagain_sparseview.shape[2]), 
                                                         dtype='float32', data=projectionsagain_sparseview[:,:,:])
                                                         
file_out_hdf5.close()

 

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


#### retrieve the projection data from Interpolated sinogram ###

projectionsagain_Interpolated=numpy.zeros((noAngles,pixres1,pixres2))

for i in range (0,noAngles):
    for j in range (0,pixres1):
        projectionsagain_Interpolated[i,j,:] = sinograminterpolated[j,i,:]


  
#plt.figure(3)
#plt.title('projectionsagain_sparseview')
#plt.imshow(projectionsagain_sparseview[0,:,:])
#plt.show()
 
print(projectionsagain_Interpolated.shape[0],projectionsagain_Interpolated.shape[1],projectionsagain_Interpolated.shape[2])





#### Create the hdf5 file for the training and label patches ####

print ("Creating HDF5 file for the Interpolated Projs")
   
file_out_hdf5=h5py.File(args.savedfile+'InterpolatedProjs.hdf5','w')
file_out_hdf5.create_dataset("Type",data=[83,105,109,112,108,101,67,111,110,101,66,101,97,109,67,84,73,
                                          109,97,103,101,83,101,113,117,101,110,99,101], shape=(29,1))
file_out_hdf5.create_dataset("Dimension", data=[65,116,116,101,110,117,97,116,105,111,110], shape = (11,1))
file_out_hdf5.create_dataset("DetectorPixelSizeX", data=PixelSizeX, shape = (1,))
file_out_hdf5.create_dataset("DetectorPixelSizeY", data=PixelSizeY, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceAxis", data=SourceAxis, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceDetector", data=SourceDetector, shape = (1,))
file_out_hdf5.create_dataset("Angle", data=angles)
images_out = file_out_hdf5.create_dataset("Image",shape=(projectionsagain_Interpolated.shape[0],
                                                         projectionsagain_Interpolated.shape[1],
                                                         projectionsagain_Interpolated.shape[2]), 
                                                         dtype='float32', data=projectionsagain_Interpolated[:,:,:])
                                                         
file_out_hdf5.close()


#### Extract patches from the interpolated sinogram ####

# Define patch size and stride
patch_size = (args.patchsize, args.patchsize)
stride = 10
patches_all_interp_siogram=numpy.zeros((10000000,args.patchsize,args.patchsize),dtype='f')


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
#checkpoint_path = "/zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/files/checkpoint_Swin2SR_1block_DownSample10_100Iter.pth"

checkpoint_path = "/zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/files/checkpoint_"+args.model+"_DownFactor10.pth" 

print('checkpoint_path is =',checkpoint_path)
#checkpoint_path = "/zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-Swin/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/files/checkpoint_UNet_DownFactor10.pth"

# Determine device and number of GPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

# Model, Optimizer, Scheduler
loader = ModelLoader()
model = loader.get_model(args.model)
    
# Setup model
#model = load_model(args.model)

#The model should moved to the default device (cuda:0) before wrapping it with nn.DataParallel.
#The input tensors are also on the default device (cuda:0).
# Move the model to the default device (cuda:0)



# Load checkpoint BEFORE wrapping with DataParallel
#model.load_state_dict(torch.load(checkpoint_path, map_location=device))



# Load the checkpoint
checkpoint = torch.load(checkpoint_path)

# Extract only the model state dict
model_state_dict = checkpoint['model_state_dict']

# Load the model state dict into your model
model.load_state_dict(model_state_dict)
print(next(model.parameters()).device)  # Check the device of the model's parameters
model = model.to(device)
model.eval()

# Wrap model with DataParallel if multiple GPUs available
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs!")
    model = nn.DataParallel(model)

#model = model.to(device)

print(next(model.parameters()).device)  # Check the device of the model's parameters
print('patches size to check is =',patches_all_interp_siogram.shape[0],patches_all_interp_siogram.shape[1],patches_all_interp_siogram.shape[2])
image = numpy.expand_dims(patches_all_interp_siogram, axis=1)
image = image.astype(numpy.float32)
xx = torch.from_numpy(image)

# Create DataLoader for efficient batching
dataset = TensorDataset(xx)
loader = DataLoader(
    dataset,
    batch_size=args.correcbatch,
    shuffle=False,
    num_workers=4,          # Adjust based on CPU cores
    pin_memory=True         # Faster data transfer to GPU
)

# Preallocate output tensor on GPU (half precision)
pred_y_all = torch.zeros(xx.shape[0], *patches_all_interp_siogram.shape[1:], 
                        dtype=torch.float16, device=device)

start_time = time.time()

with torch.no_grad():
    for batch_idx, (x,) in enumerate(loader):
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        
        #print(model.device)  # Check the device of the model
        #print('Check the device of the input tensor=',x.device)  # Check the device of the input tensor

        #print(next(model.parameters()).device) 
        # Enable mixed precision inference
        with torch.cuda.amp.autocast():
            pred_y = model(x)
            
        # Store results
        start_idx = batch_idx * args.correcbatch
        end_idx = start_idx + x.size(0)
        pred_y_all[start_idx:end_idx] = pred_y.squeeze(1).half()  # Store as half precision

# Move results to CPU if needed
pred_y_all = pred_y_all.cpu().numpy()

pred_y_all[(int(xx.size(0)/args.correcbatch)*args.correcbatch):,:,:]=patches_all_interp_siogram[(int(xx.size(0)/args.correcbatch)*args.correcbatch):,:,:]

print('fff=',int(xx.size(0)/args.correcbatch)-1,batch_idx)

total_time = time.time() - start_time
print(f'Done correcting patches in {total_time:.2f} seconds')

'''
pred_y_all=pred_y_all.reshape(patches_all_interp_siogram.shape[0]*patches_all_interp_siogram.shape[1]*patches_all_interp_siogram.shape[2])

output_file=open("/lhome/alsaffar/ctutil/PatchesAll.scan","wb")
float_array=numpy.float16(pred_y_all)
float_array.tofile(output_file)
output_file.close()
'''
#### convert the correct patches to the correct sinogram image #####

reconstructed_image_all = numpy.zeros_like(sinograminterpolated[:,:,:])

eachpatchsize=int(patches_all_interp_siogram[:,:,:].shape[0]/sinograminterpolated[:,:,:].shape[0])

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
    #reconstructed_image /= overlap_count
    
    reconstructed_image = numpy.divide(
    reconstructed_image, 
    overlap_count, 
    where=overlap_count != 0,  # Only divide where overlap_count > 0
    out=numpy.zeros_like(reconstructed_image))
    
    # Verify if the reconstruction matches the original image
    #print("Reconstruction matches original image:", numpy.allclose(sinograminterpolated[k,:,:], reconstructed_image))
    reconstructed_image_all[k,:,:]=reconstructed_image
    
    







#### retrieve the projection data from corrected sinogram ###

projectionsagain_corrected=numpy.zeros((noAngles,pixres1,pixres2))

for i in range (0,noAngles):
    for j in range (0,pixres1):
        projectionsagain_corrected[i,j,:] = reconstructed_image_all[j,i,:]



#plt.figure(3)
#plt.title('projectionsagain_corrected')
#plt.imshow(projectionsagain_corrected[0,:,:])
#plt.show()


'''
#### Create a sinogram from projection data ####
originalsinogram=numpy.zeros((pixres1,int(noAngles),pixres2))

for i in range (0,pixres1):
    for j in range (0,noAngles):
         originalsinogram[i,j,:] = RealScan[j,i,:]

'''



sinogramstreached=numpy.zeros((originalsinogram.shape[0],originalsinogram.shape[1],originalsinogram.shape[2]))

for i in range (0,noAngles,10):
    if (i<100):
       print(" i is=",i,noAngles)
    sinogramstreached[0,i,:]=originalsinogram[0,i,:]
    
plt.figure(1)
plt.axis('off')
#plt.title('Original Sinogram')
plt.imshow(originalsinogram[0,:,:],cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/originalsinogram",dpi=600,bbox_inches='tight')

plt.figure(2)
plt.axis('off')
#plt.title('Interpolated Sinogram')
plt.imshow(sinograminterpolated[0,:,:],cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/sinograminterpolated",dpi=600,bbox_inches='tight')

plt.figure(3)
plt.axis('off')
#plt.title('Sparse Sinogram')
plt.imshow(sinogramstreached[0,:,:],vmin=0,vmax=0.007,cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/sinogram",dpi=600,bbox_inches='tight')

plt.figure(4)
plt.axis('off')
#plt.title('Sparse Sinogram')
plt.imshow(reconstructed_image_all[0,:,:],cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/Correctedsinogram",dpi=600,bbox_inches='tight')

plt.figure(4)
plt.axis('off')
#plt.title('Patches')
plt.imshow(patches_all_interp_siogram[1000,:,:],cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/patches_all_interp_siogram_1",dpi=600,bbox_inches='tight')

plt.figure(5)
plt.axis('off')
#plt.title('Patches')
plt.imshow(patches_all_interp_siogram[1200,:,:],cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/patches_all_interp_siogram_2",dpi=600,bbox_inches='tight')

plt.figure(6)
plt.axis('off')
#plt.title('Patches')
plt.imshow(patches_all_interp_siogram[1500,:,:],cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/patches_all_interp_siogram_3",dpi=600,bbox_inches='tight')

plt.figure(7)
plt.axis('off')
#plt.title('Patches')
plt.imshow(patches_all_interp_siogram[1800,:,:],cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/patches_all_interp_siogram_4",dpi=600,bbox_inches='tight')

plt.figure(8)
plt.axis('off')
#plt.title('Patches')
plt.imshow(pred_y_all[1000,:,:],cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/patches_all_interp_siogram_1_Corrected",dpi=600,bbox_inches='tight')

plt.figure(9)
plt.axis('off')
#plt.title('Patches')
plt.imshow(pred_y_all[1200,:,:],cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/patches_all_interp_siogram_2_Corrected",dpi=600,bbox_inches='tight')

plt.figure(10)
plt.axis('off')
#plt.title('Patches')
plt.imshow(pred_y_all[1500,:,:],cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/patches_all_interp_siogram_3_Corrected",dpi=600,bbox_inches='tight')

plt.figure(11)
plt.axis('off')
#plt.title('Patches')
plt.imshow(pred_y_all[1800,:,:],cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/patches_all_interp_siogram_4_Corrected",dpi=600,bbox_inches='tight')

plt.figure(12)
plt.axis('off')
#plt.title('The Difference Between Original Sinogram and Interpolated One')
plt.imshow(abs(originalsinogram[0,:,:]-sinograminterpolated[0,:,:]),vmin=0,vmax=0.2,cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/diff_orgsing-interpsing",dpi=600,bbox_inches='tight')

plt.figure(13)
plt.axis('off')
#plt.title('Difference between corrected siogram and interpolated one')
plt.imshow(abs(originalsinogram[0,:,:]-reconstructed_image_all[0,:,:]),vmin=0,vmax=0.2,cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/diff_orgsing-corrsing",dpi=600,bbox_inches='tight')


plt.figure(14)
plt.axis('off')
#plt.title('Sparse Sinogram')
plt.imshow(sinogram[0,:,:],cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/sinogram2",dpi=600,bbox_inches='tight')

plt.show()
exit()

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

print('Done')
exit()






