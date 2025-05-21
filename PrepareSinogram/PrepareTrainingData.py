#python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-Swin/PrepareSinogram/PrepareTrainingData.py --projectionfile /lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/Orginal_AsAttenuation//projdownatten_R002255.hdf5  --downfactor 10 --savedfile /lhome/alsaffar/NewProjects/SparseViewInterpolation/  --ylimt1 0 --ylimt2 424 --xlimt1 0 --xlimt2 800  --patchsize 64

def arrang_array(i,data11,pix1,pix2,pixupper1,pixupper2):

  xt=numpy.linspace(0,pix2,pix2)
  yt=numpy.linspace(0,pix1,pix1) 

  X, Y=numpy.meshgrid(xt,yt)

  xt2=numpy.linspace(0,pix2,pixupper2)
  yt2=numpy.linspace(0,pix1,pixupper1)

  

  io=numpy.transpose(data11[:,:])
  
  ff1=RectBivariateSpline(xt,yt,io)
  arr_n=ff1(xt2,yt2)
  
  
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
                       help='Down factor for sparse view.')
parser.add_argument('--ylimt2', type=int, required=True,
                       help='Down factor for sparse view.')  
parser.add_argument('--xlimt1', type=int, required=True,
                       help='Down factor for sparse view.')
parser.add_argument('--xlimt2', type=int, required=True,
                       help='Down factor for sparse view.')  
parser.add_argument('--patchsize', type=int, required=True,
                       help='Down factor for sparse view.')                         
                                                                 
args = parser.parse_args()

Hdf5File=h5py.File(args.projectionfile,'r')
RealScan_NotNorm=Hdf5File["Image"][:,args.ylimt1:,args.xlimt1:args.xlimt2]
noAngles=(RealScan_NotNorm.shape[0])-1
print(RealScan_NotNorm.shape)
print('noAngles=',noAngles)
pixres1=RealScan_NotNorm.shape[1]
pixres2=RealScan_NotNorm.shape[2]

SourceAxis=Hdf5File["DistanceSourceAxis"][:]
SourceDetector=Hdf5File["DistanceSourceDetector"][:]
PixelSizeX=Hdf5File["DetectorPixelSizeX"][:]
PixelSizeY=Hdf5File["DetectorPixelSizeY"][:]
angles=Hdf5File["Angle"][:]

Hdf5File.close()



#### Normalize the data ####

maxelement=numpy.max(RealScan_NotNorm[:,:,:])
RealScan = numpy.divide(RealScan_NotNorm[:,:,:],maxelement)

print('the max value in real scan is =',numpy.max(RealScan[0,:,:]))
del RealScan_NotNorm

#### Create a sinogram from projection data
originalsinogram=numpy.zeros((pixres1,int(noAngles),pixres2))

for i in range (0,pixres1):
    for j in range (0,noAngles):
         originalsinogram[i,j,:] = RealScan[j,i,:]


'''   
plt.figure(1)
plt.title('originalsinogram')
plt.imshow(originalsinogram[0,:,:])
'''  
   
#### Extract patches from the original sinogram ####

# Define patch size and stride
patch_size = (args.patchsize, args.patchsize)
stride = 10
patches_all_org_siogram=numpy.zeros((10000000,args.patchsize,args.patchsize))


for i in range (0,originalsinogram[:,:,:].shape[0]):
    
    image=originalsinogram[i,:,:]
    
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

    
    patches_all_org_siogram[patches.shape[0]*i:patches.shape[0]*(i+1),:,:]=patches
    
    #print('list small=',patches.shape[0]*i,patches.shape[0]*(i+1))
 
    limit=patches.shape[0]*(i+1)
    
patches_all_org_siogram=patches_all_org_siogram[0:limit]

print('The size of the patches from original sinogram=',patches_all_org_siogram.shape)


'''
#### Retrieve the original image #####

reconstructed_image_all = numpy.zeros_like(originalsinogram[:,:,:])
#original_shape= (146, 33, 50, 50)

#Find each patch size
eachpatchsize=int(patches_all_org_siogram[:,:,:].shape[0]/originalsinogram[:,:,:].shape[0])

for k in range (0,originalsinogram[:,:,:].shape[0]):

    patches=patches_all_org_siogram[k*eachpatchsize:(k+1)*eachpatchsize,:,:]

    patches_ = patches.reshape(original_shape)

    # Get the shape of the patches array
    num_patches_y, num_patches_x, _, _ = patches_.shape

    # Initialize an empty array to reconstruct the image
    reconstructed_image = numpy.zeros_like(originalsinogram[k,:,:])
    overlap_count = numpy.zeros_like(originalsinogram[k,:,:])  # To count how many patches contribute to each pixel

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
    print("Reconstruction matches original image:", numpy.allclose(originalsinogram[k,:,:], reconstructed_image))
    reconstructed_image_all[k,:,:]=reconstructed_image
    
    
plt.figure(3)
plt.imshow(reconstructed_image_all[0,:,:]-originalsinogram[0,:,:])
plt.show()

exit()
'''  


#### Generate sparse view sinogram #####

sinogram=numpy.empty((pixres1,int(noAngles/args.downfactor),pixres2))

for i in range(0,pixres1):
    k=0
    for j in range (0,noAngles,args.downfactor):
        #print(j,k)
        sinogram[i,k,:]=originalsinogram[i,j,:]
        k+=1
    

#### retrieve the projection data from the sparse view sinogram (this is for display the effect of sparse view) ###

projectionsagain_sparse=numpy.zeros((int(noAngles/args.downfactor),pixres1,pixres2))

for i in range (0,int(noAngles/args.downfactor)):
    for j in range (0,pixres1):
        projectionsagain_sparse[i,j,:] = sinogram[j,i,:]



plt.figure(3)
plt.title('projectionsagain_sparse')
plt.imshow(projectionsagain_sparse[0,:,:])
plt.show()


print('angles new=',len(angles[0:noAngles:args.downfactor]))
print('angles old=',len(angles[:]))


#### Create the hdf5 file for the sparse view projections ####

print ("Create the hdf5 file for the sparse view projections")
   
file_out_hdf5=h5py.File(args.savedfile+'SparseProjection_R002255.hdf5','w')
file_out_hdf5.create_dataset("Type",data=[83,105,109,112,108,101,67,111,110,101,66,101,97,109,67,84,73,
                                          109,97,103,101,83,101,113,117,101,110,99,101], shape=(29,1))
file_out_hdf5.create_dataset("Dimension", data=[65,116,116,101,110,117,97,116,105,111,110], shape = (11,1))
file_out_hdf5.create_dataset("DetectorPixelSizeX", data=PixelSizeX, shape = (1,))
file_out_hdf5.create_dataset("DetectorPixelSizeY", data=PixelSizeY, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceAxis", data=SourceAxis, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceDetector", data=SourceDetector, shape = (1,))
file_out_hdf5.create_dataset("Angle", data=angles[0:noAngles:args.downfactor])
images_out = file_out_hdf5.create_dataset("Image",shape=(projectionsagain_sparse[:,:,:].shape[0],
                                                         projectionsagain_sparse[:,:,:].shape[1],
                                                         projectionsagain_sparse[:,:,:].shape[2]), 
                                                         dtype='float32', data=projectionsagain_sparse[:,:,:])
                                                         


file_out_hdf5.close()

print ("done creating the hdf5 file for the sparse view projections")



#### Linearly interpolate the sparse view sinogram to get it in the same size as the original sinogram ####

### Pix1 is 576 and Pix2 is 800 , pixupper1=2304 , pixupper2=3200
pix1=sinogram.shape[1]
pix2=sinogram.shape[2] 
pixupper1=noAngles
pixupper2=pixres2


print(pix1,pix2,pixupper1,pixupper2,sinogram.shape[0])

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
'''
plt.figure(1)
plt.title('Interpolated sinogram')
plt.imshow(sinograminterpolated[0,:,:])
plt.figure(2)
plt.title('the difference between original sinogram and interpolated one')
plt.imshow(abs(originalsinogram[0,:,:]-sinograminterpolated[0,:,:]))
plt.show()
'''

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



#### retrieve the projection data from the Interpolated sparse view sinogram (this is for display the effect of interpolation) ###

projectionsagain_interpolated=numpy.zeros((int(noAngles),pixres1,pixres2))

for i in range (0,int(noAngles)):
    for j in range (0,pixres1):
        projectionsagain_interpolated[i,j,:] = sinograminterpolated[j,i,:]



#plt.figure(3)
#plt.title('projectionsagain_interpolated')
#plt.imshow(projectionsagain_interpolated[0,:,:])
#plt.show()

#### Create the hdf5 file for the sparse view projections ####

print ("Create the hdf5 file for the Interpolated sparse view projections")
   
file_out_hdf5=h5py.File(args.savedfile+'InterpolatedProjection_R002255.hdf5','w')
file_out_hdf5.create_dataset("Type",data=[83,105,109,112,108,101,67,111,110,101,66,101,97,109,67,84,73,
                                          109,97,103,101,83,101,113,117,101,110,99,101], shape=(29,1))
file_out_hdf5.create_dataset("Dimension", data=[65,116,116,101,110,117,97,116,105,111,110], shape = (11,1))
file_out_hdf5.create_dataset("DetectorPixelSizeX", data=PixelSizeX, shape = (1,))
file_out_hdf5.create_dataset("DetectorPixelSizeY", data=PixelSizeY, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceAxis", data=SourceAxis, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceDetector", data=SourceDetector, shape = (1,))
file_out_hdf5.create_dataset("Angle", data=angles)
images_out = file_out_hdf5.create_dataset("Image",shape=(projectionsagain_interpolated[:,:,:].shape[0],
                                                         projectionsagain_interpolated[:,:,:].shape[1],
                                                         projectionsagain_interpolated[:,:,:].shape[2]), 
                                                         dtype='float32', data=projectionsagain_interpolated[:,:,:])
                                                         


file_out_hdf5.close()

print (" done creating the hdf5 file for the interpolated sparse view projections")


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
plt.title('Patch from Original Sinogram')
plt.imshow(patches_all_org_siogram[3500,:,:])
plt.figure(5)
plt.title('Patch from Interp Sinogram')
plt.imshow(patches_all_interp_siogram[3500,:,:])
plt.figure(6)
plt.title('The Difference Between Original Patch and Interpolated One')
plt.imshow(abs(patches_all_org_siogram[3500,:,:]-patches_all_interp_siogram[3500,:,:]))

plt.figure(7)
plt.title('projectionsagain_sparse')
plt.imshow(projectionsagain_sparse[0,:,:])
plt.figure(8)
plt.title('projectionsagain_interpolated')
plt.imshow(projectionsagain_interpolated[0,:,:])
plt.show()

print('no patches before=',patches_all_org_siogram.shape)

#patches_all_org_siogram=patches_all_org_siogram[0:patches_all_org_siogram.shape[0]:10,:,:]

print('no patches after=',patches_all_org_siogram.shape)

#### Create the hdf5 file for the training and label patches ####

print ("Creating HDF5 file from training data")
exit()   
file_out_hdf5=h5py.File(args.savedfile+'training_R002255.hdf5','w')
file_out_hdf5.create_dataset("Type",data=[83,105,109,112,108,101,67,111,110,101,66,101,97,109,67,84,73,
                                          109,97,103,101,83,101,113,117,101,110,99,101], shape=(29,1))
file_out_hdf5.create_dataset("Dimension", data=[65,116,116,101,110,117,97,116,105,111,110], shape = (11,1))
file_out_hdf5.create_dataset("DetectorPixelSizeX", data=PixelSizeX, shape = (1,))
file_out_hdf5.create_dataset("DetectorPixelSizeY", data=PixelSizeY, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceAxis", data=SourceAxis, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceDetector", data=SourceDetector, shape = (1,))
file_out_hdf5.create_dataset("Angle", data=angles)
images_out = file_out_hdf5.create_dataset("label",shape=(patches_all_org_siogram[:,:,:].shape[0],
                                                         patches_all_org_siogram[:,:,:].shape[1],
                                                         patches_all_org_siogram[:,:,:].shape[2]), 
                                                         dtype='float32', data=patches_all_org_siogram[:,:,:])
                                                         
images_out2 = file_out_hdf5.create_dataset("training",shape=(patches_all_interp_siogram[:,:,:].shape[0],
                                                         patches_all_interp_siogram[:,:,:].shape[1],
                                                         patches_all_interp_siogram[:,:,:].shape[2]), 
                                                         dtype='float32', data=patches_all_interp_siogram[:,:,:])


file_out_hdf5.close()








