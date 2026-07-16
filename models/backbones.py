import os
import torch
import torch.nn as nn
import timm

# Local imports
from .pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152

class PVT_Wrapper(nn.Module):
    def __init__(self, name, pretrained=True, pretrained_dir='./pretrained_pth/pvt/'):
        super().__init__()
        # Instantiate the correct PVT function
        pvt_fns = {
            'pvt_v2_b0': pvt_v2_b0,
            'pvt_v2_b1': pvt_v2_b1,
            'pvt_v2_b2': pvt_v2_b2,
            'pvt_v2_b3': pvt_v2_b3,
            'pvt_v2_b4': pvt_v2_b4,
            'pvt_v2_b5': pvt_v2_b5,
        }
        if name not in pvt_fns:
            raise ValueError(f"Unknown PVT variant {name}")
        
        self.backbone = pvt_fns[name]()
        
        if pretrained:
            path = os.path.join(pretrained_dir, f"{name}.pth")
            if os.path.exists(path):
                print(f"Loading pretrained weights for {name} from {path}")
                save_model = torch.load(path, map_location='cpu')
                model_dict = self.backbone.state_dict()
                state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
                model_dict.update(state_dict)
                self.backbone.load_state_dict(model_dict)
            else:
                print(f"Warning: Pretrained weight file not found at {path}. Initializing randomly.")

    def forward(self, x):
        return self.backbone(x)

def get_backbone(name, pretrained=True, pretrained_dir='./pretrained_pth/pvt/'):
    """
    Unified backbone factory. Returns (backbone_module, channel_list)
    where channel_list represents the number of channels of the stages
    from stage 4 down to stage 1, i.e., [ch_stage4, ch_stage3, ch_stage2, ch_stage1].
    """
    name = name.lower()
    
    # 1. PVT v2 Variants
    if name.startswith('pvt_v2_'):
        backbone = PVT_Wrapper(name, pretrained=pretrained, pretrained_dir=pretrained_dir)
        if name == 'pvt_v2_b0':
            channels = [256, 160, 64, 32]
        else:
            channels = [512, 320, 128, 64]
        return backbone, channels

    # 2. ResNet variants using local fallback if pretrained is handled or timm is unavailable
    resnet_fns = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
    }
    
    if name in resnet_fns:
        # We can use timm or local ResNet. Let's use timm as preferred external library backbone
        try:
            print(f"Loading {name} from timm as feature extractor...")
            # timm features_only=True returns features at reductions: stride 2, 4, 8, 16, 32.
            # We want indices corresponding to strides 4, 8, 16, 32, which are (1, 2, 3, 4) in timm's ResNet.
            backbone = timm.create_model(name, features_only=True, pretrained=pretrained, out_indices=(1, 2, 3, 4))
            # channels are returned in the order of output features: [ch_stage1, ch_stage2, ch_stage3, ch_stage4]
            # but EMCADNet decoder expects channels order [ch_stage4, ch_stage3, ch_stage2, ch_stage1]
            ch_list = list(reversed(backbone.feature_info.channels()))
            return backbone, ch_list
        except Exception as e:
            print(f"timm loading failed: {e}. Falling back to local resnet implementation.")
            backbone = resnet_fns[name](pretrained=pretrained)
            if name in ['resnet18', 'resnet34']:
                channels = [512, 256, 128, 64]
            else:
                channels = [2048, 1024, 512, 256]
            return backbone, channels

    # 3. Arbitrary timm backbones
    try:
        print(f"Attempting to load general backbone '{name}' from timm...")
        backbone = timm.create_model(name, features_only=True, pretrained=pretrained, out_indices=(1, 2, 3, 4))
        ch_list = list(reversed(backbone.feature_info.channels()))
        return backbone, ch_list
    except Exception as e:
        raise ValueError(f"Could not load backbone model '{name}': {e}")
