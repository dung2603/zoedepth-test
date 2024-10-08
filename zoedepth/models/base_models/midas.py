import torch
import torch.nn as nn
from torchvision.transforms import Normalize
import numpy as np
import sys
sys.path.append('/content/zoedepth-test/zoedepth/models/base_models')
from depth_anything.dpt import DepthAnything

def denormalize(x):
    """Reverses the imagenet normalization applied to the input.

    Args:
        x (torch.Tensor - shape(N,3,H,W)): input tensor

    Returns:
        torch.Tensor - shape(N,3,H,W): Denormalized input
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean

def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

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


class PrepForDepthAnything(object):
    def __init__(self, resize_mode="minimal", keep_aspect_ratio=True, img_size=384, do_resize=True):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        net_h, net_w = img_size
        # self.normalization = Normalize(
        #     mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.normalization = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resizer = Resize(net_w, net_h, keep_aspect_ratio=keep_aspect_ratio, ensure_multiple_of=14, resize_method=resize_mode) \
            if do_resize else nn.Identity()

    def __call__(self, x):
        return self.normalization(self.resizer(x))

class DepthCore(nn.Module):
    def __init__(self, depth_anything, trainable=True, 
    layer_names=( 'out_conv2','layer_1', 'layer_2', 'layer_3', 'layer_4','layer4_rn'),
    fetch_features=True, freeze_bn=False, keep_aspect_ratio=True, img_size=384, **kwargs):
        super().__init__()
        self.core = depth_anything
        self.output_channels = None
        self.core_out = {}
        self.trainable = trainable
        self.fetch_features = fetch_features
        self.handles = []
        self.layer_names = layer_names
        
        self.set_trainable(trainable)
        self.set_fetch_features(fetch_features)

        self.prep = PrepForDepthAnything(keep_aspect_ratio=keep_aspect_ratio, img_size=img_size, do_resize=kwargs.get('do_resize', True))

        if freeze_bn:
            self.freeze_bn()
       
        print(dir(self.core))

    def set_trainable(self, trainable):
        self.trainable = trainable
        if trainable:
            self.unfreeze()
        else:
            self.freeze()
        return self

    def set_fetch_features(self, fetch_features):
        self.fetch_features = fetch_features
        if fetch_features:
            if len(self.handles) == 0:
                self.attach_hooks(self.core.depth_head)
        else:
            self.remove_hooks()
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
            #print("Shape after prep: ", x.shape)

        with torch.set_grad_enabled(self.trainable):

            #print("Input size to Midascore", x.shape)
            rel_depth = self.core(x)
            #print(type(rel_depth))
            #print("Output from midas shape", rel_depth.shape)
            if not self.fetch_features:
                return rel_depth
        #print(self.core_out.keys())        
        out = [self.core_out[k] for k in self.layer_names]

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
        

    def attach_hooks(self, depth_head):
        if len(self.handles) > 0:
            self.remove_hooks()
        if "out_conv2" in self.layer_names:
            self.handles.append(list(depth_head.scratch.output_conv2.children())[
                                1].register_forward_hook(get_activation("out_conv2", self.core_out)))
        if "layer_4" in self.layer_names:
            self.handles.append(depth_head.scratch.refinenet4.register_forward_hook(
                get_activation("layer_4", self.core_out)))
        if "layer_3" in self.layer_names:
            self.handles.append(depth_head.scratch.refinenet3.register_forward_hook(
                get_activation("layer_3", self.core_out)))
        if "layer_2" in self.layer_names:
            self.handles.append(depth_head.scratch.refinenet2.register_forward_hook(
                get_activation("layer_2", self.core_out)))
        if "layer_1" in self.layer_names:
            self.handles.append(depth_head.scratch.refinenet1.register_forward_hook(
                get_activation("layer_1", self.core_out)))
        if "layer4_rn" in self.layer_names:
            self.handles.append(depth_head.scratch.layer4_rn.register_forward_hook(
                get_activation("layer4_rn", self.core_out)))

        return self


    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        return self

    def __del__(self):
        self.remove_hooks()

    def set_output_channels(self, model_type):
        self.output_channels = [256, 256, 256, 256, 256]
    
    @staticmethod
    def build(encoder="vitl",train_depthanything=True, use_pretrained_depth=True, fetch_features=False, freeze_bn=True, force_keep_ar=False,force_reload=False, **kwargs):
        if encoder not in DEPTH_CORE_SETTINGS:
            raise ValueError(f"Invalid model type: {encoder}. Must be one of {list(DEPTH_CORE_SETTINGS.keys())}")
    
        if "img_size" in kwargs:
            kwargs = DepthCore.parse_img_size(kwargs)
        img_size = kwargs.pop("img_size", [384, 384])
        #print("img_size", img_size)
        depth_anything =  DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder))

        #print(depth_anything)
        kwargs.update({'keep_aspect_ratio': force_keep_ar})
        depth_core = DepthCore(depth_anything,trainable=train_depthanything, fetch_features=fetch_features,
                               freeze_bn=freeze_bn, img_size=img_size, **kwargs)
        depth_core.set_output_channels(encoder)
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
            assert len(
                config['img_size']) == 2, "img_size should be a string with comma separated img_size=H,W"
        elif isinstance(config['img_size'], int):
            config['img_size'] = [config['img_size'], config['img_size']]
        else:
            assert isinstance(config['img_size'], list) and len(
                config['img_size']) == 2, "img_size should be a list of H,W"
        return config

 

# Model name to number of output channels
nchannels2models = {
    tuple([256]*3): ["vits", "vitb", "vitl"]
}

DEPTH_CORE_SETTINGS = {m: k for k, v in nchannels2models.items() for m in v}   
# Example usage
