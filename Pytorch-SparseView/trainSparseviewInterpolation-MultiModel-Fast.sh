#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gpus=rtx4090:3


#srun --unbuffered python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/train-Fast.py --trainingdata /lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/training_porschestators_DownFactor10_CompleteData.hdf5 --model UNet

#srun --unbuffered python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/train-Fast.py --trainingdata /lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/training_porschestators_DownFactor10.hdf5 --model ResNet

#srun --unbuffered python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/train-Fast.py --trainingdata /lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/training_porschestators_DownFactor10.hdf5 --model ODConvSR-9-96

#srun --unbuffered python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/train-Fast.py --trainingdata /lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/training_porschestators_DownFactor10.hdf5 --model UNet_Org

# --resume /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/files/checkpoint_Swin2SR_2block.pth

#srun --unbuffered python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/train-Fast.py --trainingdata /lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/training_porschestators_DownFactor10.hdf5 --model VDSR

#srun --unbuffered python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/train-Fast.py --trainingdata /lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/training_porschestators_DownFactor10.hdf5 --model SwinUnet 

#--resume /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/files/checkpoint_DRRN.pth

srun --unbuffered python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/PrepareSinogram/AllTheCorrectionProcess-Fast-Multi_test.py --projectionfile /lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/Orginal_AsAttenuation/projdownatten_R002505.hdf5  --downfactor 10 --savedfile /lhome/alsaffar/NewProjects/SparseViewInterpolation/  --ylimt1 0 --ylimt2 292 --xlimt1 0 --xlimt2 800  --patchsize 64 --correcbatch 1200 --model DRRN

#srun python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/PrepareSinogram/AllTheCorrectionProcess-Fast-Multi.py --projectionfile /lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/Orginal_AsAttenuation/projdownatten_R002505.hdf5  --downfactor 10 --savedfile /lhome/alsaffar/NewProjects/SparseViewInterpolation/  --ylimt1 0 --ylimt2 292 --xlimt1 0 --xlimt2 800  --patchsize 64 --correcbatch 1200 --model ODConvSR-9-96


#srun --unbuffered python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/PrepareSinogram/AllTheCorrectionProcess-Fast-Multi.py --projectionfile /lhome/alsaffar/NewProjects/SparseViewInterpolation/HairPins/Orginal_AsAttenuation/projdownatten_R002505.hdf5  --downfactor 10 --savedfile /lhome/alsaffar/NewProjects/SparseViewInterpolation/  --ylimt1 0 --ylimt2 292 --xlimt1 0 --xlimt2 800  --patchsize 64 --correcbatch 1200 --model UNet

#srun --unbuffered --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=32 --mem=200G  python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-Swin/PrepareSinogram/AllTheCorrectionProcess_FromPatchesDirectly.py --projectionfile /lhome/alsaffar/NewProjects/SparseViewInterpolation/training_porschestators_DownFactor20.hdf5  --model UNet

#srun --unbuffered python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/PrepareSinogram/JoinAllTrainingData.py 

