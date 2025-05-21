#python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/PrepareSinogram/ShowCorrectionEffect.py --model UNet
import os
import numpy as np
import h5py
import sys
import math,array,struct
import matplotlib.pyplot as plt
from functools import partial
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
args = parser.parse_args()

#Swin1Block
#UNet
#ResNet
## volume 1 the original  ##

volumeHDF5=h5py.File('/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002505/R002505_DownFactor10/Ctutil_OriginalProjection_R002505/volume.hdf5','r')
volume_org=volumeHDF5["Volume"][:,:,:]
gridSpacing=volumeHDF5["GridSpacing"][:]
gridOrigin=volumeHDF5["GridOrigin"][:]
#del volumedownHDF5
volumeHDF5.close()


## volume 2 the Interpolated  ##

volumeHDF5=h5py.File('/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002505/R002505_DownFactor10/Ctutil_InterpolatedProjection_R002505/volume.hdf5','r')
volume_Interp=volumeHDF5["Volume"][:,:,:]
gridSpacing=volumeHDF5["GridSpacing"][:]
gridOrigin=volumeHDF5["GridOrigin"][:]
#del volumedownHDF5
volumeHDF5.close()


## volume 3 the Sparse view  ##

volumeHDF5=h5py.File('/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002505/R002505_DownFactor10/Ctutil_SparseView_R002505/volume.hdf5','r')
volume_Sparse=volumeHDF5["Volume"][:,:,:]
gridSpacing=volumeHDF5["GridSpacing"][:]
gridOrigin=volumeHDF5["GridOrigin"][:]
#del volumedownHDF5
volumeHDF5.close()


## volume 4 the Corrected  ##

volumeHDF5=h5py.File('/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002505/R002505_DownFactor10/CorrectionResults_100Iter_Updated/'+args.model+'/Ctutil_CorrectedProjections/volume.hdf5','r')
volume_Correct=volumeHDF5["Volume"][:,:,:]
gridSpacing=volumeHDF5["GridSpacing"][:]
gridOrigin=volumeHDF5["GridOrigin"][:]
#del volumedownHDF5
volumeHDF5.close()

Diff_Org_Spars=abs(volume_org[:,30,:]-volume_Sparse[:,30,:])
Diff_Org_Interp=abs(volume_org[:,30,:]-volume_Interp[:,30,:])
Diff_Org_Correct=abs(volume_org[:,30,:]-volume_Correct[:,30,:])














ssim_ = ssim(volume_org[:,30,:], volume_Sparse[:,30,:], data_range=volume_Sparse[:,30,:].max() - volume_Sparse[:,30,:].min())

psnr_ = psnr(volume_org[:,30,:], volume_Sparse[:,30,:], data_range=volume_Sparse[:,30,:].max() - volume_Sparse[:,30,:].min())

print('SSIM and PSNR of Sparse=',ssim_,psnr_)



ssim_ = ssim(volume_org[:,30,:], volume_Interp[:,30,:], data_range=volume_Interp[:,30,:].max() - volume_Interp[:,30,:].min())

psnr_ = psnr(volume_org[:,30,:], volume_Interp[:,30,:], data_range=volume_Interp[:,30,:].max() - volume_Interp[:,30,:].min())

print('SSIM and PSNR of Interpolated=',ssim_,psnr_)

ssim_ = ssim(volume_org[:,30,:], volume_Correct[:,30,:], data_range=volume_Correct[:,30,:].max() - volume_Correct[:,30,:].min())

psnr_ = psnr(volume_org[:,30,:], volume_Correct[:,30,:], data_range=volume_Correct[:,30,:].max() - volume_Correct[:,30,:].min())


print('SSIM and PSNR of Correctd=',ssim_,psnr_)

vmin_=-5
vmax_=20

plt.figure(1)
#plt.title('Original')
plt.axis('off')
#plt.plot([400, 400],[0,791], 'w--', linewidth=1.5)
plt.imshow(volume_org[:,30,:],vmax=vmax_,vmin=vmin_,cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002505/R002505_DownFactor10/CorrectionResults_100Iter_Updated/Results/Original_"+args.model,dpi=600,bbox_inches='tight')

plt.figure(2)
#plt.title('Interpolated')
plt.axis('off')
#plt.plot([400, 400],[0,791], 'w--', linewidth=1.5)
plt.imshow(volume_Interp[:,30,:],vmax=vmax_,vmin=vmin_,cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002505/R002505_DownFactor10/CorrectionResults_100Iter_Updated/Results/Interpolated_"+args.model,dpi=600,bbox_inches='tight')

plt.figure(3)
#plt.title('SparseView')
plt.axis('off')
#plt.plot([400, 400],[0,791], 'w--', linewidth=1.5)
plt.imshow(volume_Sparse[:,30,:],vmax=vmax_,vmin=vmin_,cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002505/R002505_DownFactor10/CorrectionResults_100Iter_Updated/Results/SparseView_"+args.model,dpi=600,bbox_inches='tight')

plt.figure(4)
#plt.title('Corrected')
plt.axis('off')
#plt.plot([400, 400],[0,791], 'w--', linewidth=1.5)
plt.imshow(volume_Correct[:,30,:],vmax=vmax_,vmin=vmin_,cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002505/R002505_DownFactor10/CorrectionResults_100Iter_Updated/Results/Corrected_"+args.model,dpi=600,bbox_inches='tight')

plt.figure(5)
#plt.title('The Difference Between Original and Sparse One')
plt.axis('off')
plt.plot([0,791],[416, 416], 'w--', linewidth=1.5)
plt.imshow(Diff_Org_Spars,vmax=5,vmin=0,cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002505/R002505_DownFactor10/CorrectionResults_100Iter_Updated/Results/DiffOrgVsSparse_"+args.model,dpi=600,bbox_inches='tight')

plt.figure(6)
#plt.title('Difference between Original and interpolated one')
plt.axis('off')
plt.plot([0,791],[416, 416], 'w--', linewidth=1.5)
plt.imshow(Diff_Org_Interp,vmax=5,vmin=0,cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002505/R002505_DownFactor10/CorrectionResults_100Iter_Updated/Results/DiffOrgVsInterpolated_"+args.model,dpi=600,bbox_inches='tight')

plt.figure(7)
#plt.title('Difference between Original and Corrected one')
plt.axis('off')
plt.plot([0,791],[416, 416], 'w--', linewidth=1.5)
plt.imshow(Diff_Org_Correct,vmax=5,vmin=0,cmap='gray')
plt.savefig("/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002505/R002505_DownFactor10/CorrectionResults_100Iter_Updated/Results/DiffOrgVsCorrected_"+args.model,dpi=600,bbox_inches='tight')


t1=np.arange(0,Diff_Org_Spars.shape[1])

plt.figure(8)
a,=plt.plot(t1,Diff_Org_Spars[416,:],"b",linewidth=2.0)## for cylinder head and titanium rod it was 263
b,=plt.plot(t1,Diff_Org_Interp[416,:],"r--",linewidth=2.0)
c,=plt.plot(t1,Diff_Org_Correct[416,:],"k",linewidth=2.0)
#d,=plt.plot(t1,MyIntensityArion2[0,int(340),:],"k",linewidth=2.0)
plt.legend([a, b,c], ['Diff_Org_Spars', 'Diff_Org_Interp', 'Diff_Org_Correct'],prop={'size': 16},loc='upper left')
plt.grid()
#ply.title('Central Profile Line Between Three Projections')
plt.xlabel ('Pixels')
plt.ylabel ('Differences')
plt.savefig("/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002505/R002505_DownFactor10/CorrectionResults_100Iter_Updated/Results/Profile_"+args.model,dpi=600,bbox_inches='tight')

plt.figure(9)
a,=plt.plot(t1,Diff_Org_Spars[416,:],"b",linewidth=2.0)## for cylinder head and titanium rod it was 263

#d,=plt.plot(t1,MyIntensityArion2[0,int(340),:],"k",linewidth=2.0)
plt.legend([a], ['Diff_Org_Spars'],prop={'size': 16},loc='upper left')
plt.grid()
#ply.title('Central Profile Line Between Three Projections')
plt.xlabel ('Pixels')
plt.ylabel ('Differences')
plt.savefig("/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002505/R002505_DownFactor10/CorrectionResults_100Iter_Updated/Results/Profile_Diff_Org_Spars_"+args.model,dpi=600,bbox_inches='tight')

plt.figure(10)
a,=plt.plot(t1,Diff_Org_Interp[416,:],"r--",linewidth=2.0)
#d,=plt.plot(t1,MyIntensityArion2[0,int(340),:],"k",linewidth=2.0)
plt.legend([a], ['Diff_Org_Interp'],prop={'size': 16},loc='upper left')
plt.grid()
#ply.title('Central Profile Line Between Three Projections')
plt.xlabel ('Pixels')
plt.ylabel ('Differences')
plt.savefig("/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002505/R002505_DownFactor10/CorrectionResults_100Iter_Updated/Results/Profile_Diff_Org_Interp_"+args.model,dpi=600,bbox_inches='tight')

plt.figure(11)
a,=plt.plot(t1,Diff_Org_Correct[416,:],"k",linewidth=2.0)
#d,=plt.plot(t1,MyIntensityArion2[0,int(340),:],"k",linewidth=2.0)
plt.legend([a], ['Diff_Org_Correct'],prop={'size': 16},loc='upper left')
plt.grid()
#ply.title('Central Profile Line Between Three Projections')
plt.xlabel ('Pixels')
plt.ylabel ('Differences')
plt.savefig("/lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/R002505/R002505_DownFactor10/CorrectionResults_100Iter_Updated/Results/Profile_Diff_Org_Correct_"+args.model,dpi=600,bbox_inches='tight')
plt.show()





