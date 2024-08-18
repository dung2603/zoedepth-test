# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import Normalize,InterpolationMode
import requests
import os
from depth_anything.dpt import DPT_DINOv2
def denormalize(self, x):
        """Denormalize images to the original range."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = x * std + mean
        return x
class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
    ):
        """Init.
        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        print("Params passed to Resize transform:")
        print("\twidth: ", width)
        print("\theight: ", height)
        print("\tresize_target: ", resize_target)
        print("\tkeep_aspect_ratio: ", keep_aspect_ratio)
        print("\tensure_multiple_of: ", ensure_multiple_of)
        print("\tresize_method: ", resize_method)

        self.__width = width
        self.__height = height

        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(
                f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, x):
        width, height = self.get_size(*x.shape[-2:][::-1])
        return nn.functional.interpolate(x, (height, width), mode='bilinear', align_corners=True)
class PrepForDepth(object):
    def __init__(self, resize_mode="minimal", keep_aspect_ratio=True, img_size=384, do_resize=True):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        net_h, net_w = img_size
        self.normalization = Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.resizer = Resize((net_h, net_w), interpolation=InterpolationMode.BILINEAR) \
            if do_resize else nn.Identity()

    def __call__(self, x):
        x = self.resizer(x)
        x = self.normalization(x)
        return x
class DepthCore(nn.Module):
    def __init__(self, depth_model, trainable=False, fetch_features=True, layer_names=('output_conv1', 'layer1_rn', 'layer2_rn', 'layer3_rn', 'layer4_rn'), freeze_bn=False, keep_aspect_ratio=True, img_size=384, **kwargs):
        super().__init__()
        self.core = depth_model
        self.trainable = trainable
        self.fetch_features = fetch_features
        self.layer_names = layer_names

        self.set_trainable(trainable)

        self.prep = PrepForDepth(keep_aspect_ratio=keep_aspect_ratio, img_size=img_size, do_resize=kwargs.get('do_resize', True))

        if freeze_bn:
            self.freeze_bn()

    def set_trainable(self, trainable):
        self.trainable = trainable
        if trainable:
            self.unfreeze()
        else:
            self.freeze()
        return self

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.trainable = False
        return self

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.trainable = True
        return self

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x, denorm=False, return_rel_depth=False):
        with torch.no_grad():
            if denorm:
                x = denormalize(x)
            x = self.prep(x)

        with torch.set_grad_enabled(self.trainable):
            rel_depth = self.core(x)
            if not self.fetch_features:
                return rel_depth

        out = [getattr(self.core.scratch, name) for name in self.layer_names]

        if return_rel_depth:
            return rel_depth, out
        return out

    def get_rel_pos_params(self):
        for name, p in self.core.pretrained.named_parameters():
            if "relative_position" in name:
                yield p

    def get_enc_params_except_rel_pos(self):
        for name, p in self.core.pretrained.named_parameters():
            if "relative_position" not in name:
                yield p

    def freeze_encoder(self, freeze_rel_pos=False):
        if freeze_rel_pos:
            for p in self.core.pretrained.parameters():
                p.requires_grad = False
        else:
            for p in self.get_enc_params_except_rel_pos():
                p.requires_grad = False
        return self
    
    def set_output_channels(self, model_type):
        self.output_channels = DEPTH_CORE_SETTINGS[model_type]
    
# Định nghĩa hàm build
    def build(depth_model_type="Depth-Anything-Large", train_depth=False, use_pretrained_depth=True, fetch_features=False, freeze_bn=True, force_keep_ar=False, force_reload=False, **kwargs):
        if depth_model_type not in DEPTH_CORE_SETTINGS:
            raise ValueError(f"Invalid model type: {depth_model_type}. Must be one of {list(DEPTH_CORE_SETTINGS.keys())}")

        if "img_size" in kwargs:
            kwargs = DepthCore.parse_img_size(kwargs)
        img_size = kwargs.pop("img_size", [384, 384])
        print("img_size", img_size)
        if use_pretrained_depth:
           model_url = MODEL_URLS[depth_model_type]
           model_name = depth_model_type

        # Download the model if it doesn't already exist
           model_path = download_model(model_url, model_name)

        # Load the model from the local path
           state_dict = torch.load(model_path)
           depth_model = DPT_DINOv2()  # Khởi tạo mô hình của bạn
           depth_model.load_state_dict(state_dict)  # Tải trọng số vào mô hình
        else:
        # Load the model from the provided file
           depth_model  = torch.load("dpt.py")

        kwargs.update({'keep_aspect_ratio': force_keep_ar})
        depth_core = DepthCore(depth_model, trainable=train_depth, fetch_features=fetch_features,
                               freeze_bn=freeze_bn, img_size=img_size, **kwargs)
        depth_core.set_output_channels(depth_model_type)
        return depth_core

    @staticmethod
    def build_from_config(config):
        return DepthCore.build(**config)

    @staticmethod
    def parse_img_size(config):
        assert 'img_size' in config
        if isinstance(config['img_size'], str):
            assert "," in config['img_size'], "img_size should be a string with comma separated img_size=H,W"
            config['img_size'] = list(map(int, config['img_size'].split(",")))
            assert len(config['img_size']) == 2, "img_size should be a string with comma separated img_size=H,W"
        elif isinstance(config['img_size'], int):
            config['img_size'] = [config['img_size'], config['img_size']]
        else:
            assert isinstance(config['img_size'], list) and len(config['img_size']) == 2, "img_size should be a list of H,W"
        return config

def download_model(url, model_name):
    model_path = f"{model_name}.pth"
    if not os.path.exists(model_path):
        print(f"Downloading {model_name} model...")
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print(f"{model_name} model downloaded and saved as {model_path}.")
    else:
        print(f"{model_name} model already exists at {model_path}.")
    return model_path

# URLs for the models
MODEL_URLS = {
    "Depth-Anything-Small": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
    "Depth-Anything-Base": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true",
    "Depth-Anything-Large": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
}

# Mapping number of output channels to model names
nchannels2models = {
    tuple([256]*3): ["Depth-Anything-Small", "Depth-Anything-Base", "Depth-Anything-Large"]
}

# Model name to number of output channels
DEPTH_CORE_SETTINGS = {m: k for k, v in nchannels2models.items() for m in v}



