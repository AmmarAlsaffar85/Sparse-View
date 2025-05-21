import os.path
import torch

#from models.AsConvSR_V0 import AsConvSR as AsConvSR_V0
#from models.AsConvSR_V1 import AsConvSR as AsConvSR_V1
#from models.CombinedModule import CombinedModule
from models.ODConvSR import ODConvSR
from models.Swin2SR import Swin2SR
from models.UNet import UNet
from models.UNet_Org import UNet_Org
from models.ResConvSR import ResConvSR
from models.SRResNet import SRResNet
from models.VDSR import VDSR
from models.DRRN import DRRN
from models.vision_transformer import SwinUnet

class ModelLoader:
    def __init__(self):
        self.image_size = 64

    def get_model(self, model_name: str) -> torch.nn.Module:
        if model_name == 'Swin2SR_1block':
            return Swin2SR(img_size=self.image_size, embed_dim=64, depths=(4,), num_heads=(4,), window_size=8, mlp_ratio=2.)
        elif model_name == 'Swin2SR_2block':
            return Swin2SR(img_size=self.image_size, embed_dim=64, depths=(4, 4), num_heads=(4, 4), window_size=8,mlp_ratio=2.)
        elif model_name == 'Swin2SR':
            return Swin2SR(img_size=self.image_size, embed_dim=64, depths=(4, 4, 4, 4), num_heads=(4, 4, 4, 4),mlp_ratio=2., window_size=8)
        elif model_name == 'UNet':
            return UNet()
        elif model_name == 'UNet_Org':
            return UNet_Org()
        elif model_name == 'ODConvSR-9-96':
            return ODConvSR(odconv_blocks=9, embed_dim=96)
        elif model_name == 'ResNet':
            return SRResNet(in_channels=1, out_channels=1, channels=64, num_rcb=16, upscale=1)
        elif model_name == 'VDSR':
            return VDSR()
        elif model_name == 'DRRN':
            return DRRN(num_residual_unit=8)
        elif model_name == 'SwinUnet':
            return SwinUnet(img_size=self.image_size, embed_dim=64, depths=(4,4,), depths_decoder=[4,4,], num_heads=(4,4,), mlp_ratio=2., window_size=8) 
        # Add more models as needed
        else:
            raise ValueError(f"Unknown model: {model_name}")

##### Notes #####

#- Swin2SR_1block  uses 1 RSTB with 4 Swin Transformer Layers (S2TL). This is very shallow, but fits in the 24GB GPU memory.
#- Swin2SR_2block  uses 2 RSTB, 4 S2TL each
#- Swin2SR uses 4 RSTB, each 4 S2TL. GPU memory is the problem here.
#- For comparison, the default config in the original code is 4 RSTB, 6 S2TL each.
# See Paper Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration (Fig.1). Swin2SR_1block  uses 1 RSTB with 4 Swin Transformer Layers (S2TL), this is referred
# in the code as depths=(4,). The num_heads=(4,) means every layer (S2TL) has 4 attention heads 
# Swin2SR_2block  uses 2 RSTB with 4 Swin Transformer Layers (S2TL), this is referred
# in the code as depths=(4,4). The num_heads=(4,) means every layer (S2TL) has 4 attention heads 
'''
def load_model(model_name: str) -> torch.nn.Module:
    models = load_models()
    #print(models[model_name])
    if model_name not in models.keys():
        raise ValueError(f'Unknown model {model_name}')
    return models[model_name]
'''


def load_pretrained_models(device=torch.device('cpu')) -> dict[str, torch.nn.Module]:
    pretrained_models = {}
    models = load_models( )
    for model_name, model in models.items():
        model_path = f'model_zoo/_x{scale_factor}/{model_name}/{model_name}.pth' # just loading
        if os.path.isfile(model_path):
            model_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict=model_dict, strict=True)
            pretrained_models[model_name] = model.to(device)
    return pretrained_models


def load_pretrained_model(model_name: str,
                          device=torch.device('cpu')) -> torch.nn.Module:
    pretrained_models = load_pretrained_models(device)
    if model_name not in pretrained_models.keys():
        raise ValueError(f'Unknown model {model_name}')
    return pretrained_models[model_name]


def implements_temperature_annealing(model_name: str) -> bool:
    """
    Check if the model is capable of temperature decay. If implemented, temperature decay will decay the temperature of
    softmax computation. It should be used when training models with dynamic convolution.

    Args:
        model_name: name of the model.

    Returns:
        whether temperature decay is implemented for model.
    """
    # Models AsConvSR from V2 upward support temperature decay.
    return model_name.startswith('AsConvSR_V1') or model_name.startswith('ODConvSR')



def output_segmentation(dataset_type: str):
    return dataset_type == 'combined_dataset'
    

'''
def dataset_paths(dataset_type: str, scale_factor: int) -> tuple[str, str, str]:
    datasets_path ='/lhome/alsaffar/NewProjects/SuperRes/NewDataPoresInsertion/TrainingDataSet'
    if dataset_type == 'simulated_dataset' or dataset_type == 'binary_dataset' or dataset_type == 'combined_dataset':
        return f'{datasets_path}/simulated_x{scale_factor}/', \
            f'{datasets_path}/simulated_validation_hairpins_x{scale_factor}', \
            f'{datasets_path}/simulated_test_hairpins_x{scale_factor}'
    elif dataset_type == 'combined_dataset':
        return f'{datasets_path}/simulated_segmented_training_hairpins_x{scale_factor}', \
            f'{datasets_path}/simulated_segmented_validation_hairpins_x{scale_factor}', \
            f'{datasets_path}/simulated_segmented_test_hairpins_x{scale_factor}'
    elif dataset_type == 'pores_dataset':
        return f'{datasets_path}/simulated_training_pores_x{scale_factor}', \
            f'{datasets_path}/simulated_validation_pores_x{scale_factor}', \
            f'{datasets_path}/simulated_test_pores_x{scale_factor}'
    elif dataset_type == 'RWTH_dataset':
        return f'{datasets_path}/RWTH_training_hairpins_x{scale_factor}', \
            f'{datasets_path}/RWTH_validation_hairpins_x{scale_factor}', \
            ''
    else:
        raise ValueError(f'Invalid dataset type. Must be "simulated_dataset", "combined_dataset", "pores_dataset",'
                         f'"binary_dataset" or "RWTH_dataset", is {dataset_type}.')
'''
'''
def training_dataset_path(dataset_type: str, scale_factor: int) -> str:
    return dataset_paths(dataset_type, scale_factor)[0]


def validation_dataset_path(dataset_type: str, scale_factor: int) -> str:
    return dataset_paths(dataset_type, scale_factor)[1]


def test_dataset_path(dataset_type: str, scale_factor: int) -> str:
    return dataset_paths(dataset_type, scale_factor)[2]
'''
