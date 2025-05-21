import os
import numpy as np
import h5py
import sys
import math,array,struct
import matplotlib.pyplot as plt
from functools import partial
from scipy import ndimage

### Read the Projections 1 ###

## Projections file 1  ##

Hdf5File=h5py.File('/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002174/R002174_DownFactor10/training_R002174.hdf5','r')

R002174_training=Hdf5File["training"][:,:,:]
R002174_label=Hdf5File["label"][:,:,:]

noAngles=(R002174_training.shape[0])
print(R002174_training.shape)
print('noAngles=',noAngles)
pixres1=R002174_training.shape[1]
pixres2=R002174_training.shape[2]

SourceAxis=Hdf5File["DistanceSourceAxis"][:]
SourceDetector=Hdf5File["DistanceSourceDetector"][:]
PixelSizeX=Hdf5File["DetectorPixelSizeX"][:]
PixelSizeY=Hdf5File["DetectorPixelSizeY"][:]
angles=Hdf5File["Angle"][:]

Hdf5File.close()

print(" ... R002174_training size: ",R002174_training.shape)
print(" ... R002174_label size: ",R002174_label.shape)


##########################################################################################

## Projections file 2  ##

Hdf5File=h5py.File('/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002176/R002176_DownFactor10/training_R002176.hdf5','r')

R002176_training=Hdf5File["training"][:,:,:]
R002176_label=Hdf5File["label"][:,:,:]

noAngles=(R002176_training.shape[0])-1
print(R002176_training.shape)
print('noAngles=',noAngles)
pixres1=R002176_training.shape[1]
pixres2=R002176_training.shape[2]

SourceAxis=Hdf5File["DistanceSourceAxis"][:]
SourceDetector=Hdf5File["DistanceSourceDetector"][:]
PixelSizeX=Hdf5File["DetectorPixelSizeX"][:]
PixelSizeY=Hdf5File["DetectorPixelSizeY"][:]
angles=Hdf5File["Angle"][:]

Hdf5File.close()

print(" ... R002176_training size: ",R002176_training.shape)
print(" ... R002176_label size: ",R002176_label.shape)
#########################################################################################

## Projections file 3  ##

Hdf5File=h5py.File('/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002321/R002321_DownFactor10/training_R002321.hdf5','r')

R002321_training=Hdf5File["training"][:,:,:]
R002321_label=Hdf5File["label"][:,:,:]

noAngles=(R002321_training.shape[0])-1
print(R002321_training.shape)
print('noAngles=',noAngles)
pixres1=R002321_training.shape[1]
pixres2=R002321_training.shape[2]

SourceAxis=Hdf5File["DistanceSourceAxis"][:]
SourceDetector=Hdf5File["DistanceSourceDetector"][:]
PixelSizeX=Hdf5File["DetectorPixelSizeX"][:]
PixelSizeY=Hdf5File["DetectorPixelSizeY"][:]
angles=Hdf5File["Angle"][:]

Hdf5File.close()

print(" ... R002321_training size: ",R002321_training.shape)
print(" ... R002321_label size: ",R002321_label.shape)
#########################################################################################


## Projections file 4  ##

Hdf5File=h5py.File('/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002323/R002323_DownFactor10/training_R002323.hdf5','r')

R002323_training=Hdf5File["training"][:,:,:]
R002323_label=Hdf5File["label"][:,:,:]

noAngles=(R002323_training.shape[0])-1
print(R002323_training.shape)
print('noAngles=',noAngles)
pixres1=R002323_training.shape[1]
pixres2=R002323_training.shape[2]

SourceAxis=Hdf5File["DistanceSourceAxis"][:]
SourceDetector=Hdf5File["DistanceSourceDetector"][:]
PixelSizeX=Hdf5File["DetectorPixelSizeX"][:]
PixelSizeY=Hdf5File["DetectorPixelSizeY"][:]
angles=Hdf5File["Angle"][:]

Hdf5File.close()

print(" ... R002323_training size: ",R002323_training.shape)
print(" ... R002323_label size: ",R002323_label.shape)
#########################################################################################


####################### Concatenate all the patches and label ###########################


print ('check after trancating the data training_R002174=',R002174_training.shape[0],R002174_label[:,:,:].shape[0])
print ('check after trancating the data training_R002176=',R002176_training.shape[0],R002176_label[:,:,:].shape[0])
print ('check after trancating the data training_R002321=',R002321_training.shape[0],R002321_label[:,:,:].shape[0])
print ('check after trancating the data training_R002323=',R002323_training.shape[0],R002323_label[:,:,:].shape[0])




Training_all=np.concatenate([R002174_training[0:R002174_training.shape[0]:3,:,:],R002176_training[0:R002176_training.shape[0]:3,:,:],
                             R002321_training[0:R002321_training.shape[0]:3,:,:],R002323_training[0:R002323_training.shape[0]:3,:,:]])
Label_all=np.concatenate([R002174_label[0:R002174_label.shape[0]:3,:,:],R002176_label[0:R002176_label.shape[0]:3,:,:],
                          R002321_label[0:R002321_label.shape[0]:3,:,:],R002323_label[0:R002323_label.shape[0]:3,:,:]])                             




print(" ... Training_all size: ", Training_all.shape)
print(" ... label_all size: ", Label_all.shape)



#### Create the hdf5 file for the training and label patches ####

print ("Creating HDF5 file from training data")
   
file_out_hdf5=h5py.File('/lhome/alsaffar/NewProjects/SparseViewInterpolation/training_porschestators_CompleteData.hdf5','w')
file_out_hdf5.create_dataset("Type",data=[83,105,109,112,108,101,67,111,110,101,66,101,97,109,67,84,73,
                                          109,97,103,101,83,101,113,117,101,110,99,101], shape=(29,1))
file_out_hdf5.create_dataset("Dimension", data=[65,116,116,101,110,117,97,116,105,111,110], shape = (11,1))
file_out_hdf5.create_dataset("DetectorPixelSizeX", data=PixelSizeX, shape = (1,))
file_out_hdf5.create_dataset("DetectorPixelSizeY", data=PixelSizeY, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceAxis", data=SourceAxis, shape = (1,))
file_out_hdf5.create_dataset("DistanceSourceDetector", data=SourceDetector, shape = (1,))
file_out_hdf5.create_dataset("Angle", data=angles)
images_out = file_out_hdf5.create_dataset("label",shape=(Label_all[:,:,:].shape[0],
                                                         Label_all[:,:,:].shape[1],
                                                         Label_all[:,:,:].shape[2]), 
                                                         dtype='float32', data=Label_all[:,:,:])
                                                         
images_out2 = file_out_hdf5.create_dataset("training",shape=(Training_all[:,:,:].shape[0],
                                                         Training_all[:,:,:].shape[1],
                                                         Training_all[:,:,:].shape[2]), 
                                                         dtype='float32', data=Training_all[:,:,:])


file_out_hdf5.close()



#/zhome/alsaffar/ctutil/./ctutil generate-vgi /import/scratch/tmp-ct-3/Ammar/PviotPointProject/volumes/VolumeAllPinsAlligned-Updated/volume-All_NewNormalization.hdf5

