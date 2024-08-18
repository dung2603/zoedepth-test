import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np


KEY_OUTPUT = 'metric_depth'

def extract_key(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction


class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            target = target[mask]

        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            alpha = 1e-7
            g = torch.log(input + alpha) - torch.log(target + alpha)

            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

            loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        if not return_interpolated:
            return loss

        return loss, intr_input


def grad(x):
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    mag = diff_x**2 + diff_y**2
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle


def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]


class GradL1Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        grad_gt = grad(target)
        grad_pred = grad(input)
        mask_g = grad_mask(mask)

        loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
        loss = loss + \
            nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])
        if not return_interpolated:
            return loss
        return loss, intr_input


class BerHuLoss(torch.nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, pred, target):
        diff = torch.abs(target - pred)
        delta = self.threshold * torch.max(diff).item()

        mask = diff <= delta
        l1_loss = diff[mask]
        l2_loss = (diff[~mask]**2 + delta**2) / (2 * delta)

        loss = torch.cat((l1_loss, l2_loss))
        return torch.mean(loss)

class MultiScaleGradientLoss(torch.nn.Module):
    def __init__(self, scales=[1, 2, 4]):
        super(MultiScaleGradientLoss, self).__init__()
        self.scales = scales
    def compute_gradient(self, tensor):
        dx = tensor[:, :, :, :-1] - tensor[:, :, :, 1:]
        dy = tensor[:, :, :-1, :] - tensor[:, :, 1:, :]
        return dx, dy
    def forward(self, output, target):
        loss = 0
        for scale in self.scales:
            if scale > 1:
                output_scaled = F.interpolate(output, scale_factor=1/scale, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, scale_factor=1/scale, mode='bilinear', align_corners=False)
            else:
                output_scaled = output
                target_scaled = target

            grad_output_x, grad_output_y = self.compute_gradient(output_scaled)
            grad_target_x, grad_target_y = self.compute_gradient(target_scaled)

            loss += F.mse_loss(grad_output_x, grad_target_x) + F.mse_loss(grad_output_y, grad_target_y)

        return loss
    
class PhotometricLoss(torch.nn.Module):
    def __init__(self):
        super(PhotometricLoss, self).__init__()

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        diff = torch.sum(diff, dim=1)
        return torch.mean(diff)
    
class OrdinalRegressionLoss(object):

    def __init__(self, ord_num, beta, discretization="SID"):
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, gt):
        N,one, H, W = gt.shape

        ord_c0 = torch.ones(N, self.ord_num, H, W).to(gt.device)
        if self.discretization == "SID":
            label = self.ord_num * torch.log(gt) / np.log(self.beta)
        else:
            label = self.ord_num * (gt - 1.0) / (self.beta - 1.0)
        label = label.long()
        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False) \
            .view(1, self.ord_num, 1, 1).to(gt.device)
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        mask = (mask > label)
        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0
        ord_label = torch.cat((ord_c0, ord_c1), dim=1)
        return ord_label, mask

    def __call__(self, prob, gt):
        valid_mask = gt > 0.
        ord_label, mask = self._create_ord_label(gt)
        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask.squeeze(1)]
        return loss.mean()


class DiscreteNLLLoss(nn.Module):
    """Cross entropy loss"""
    def __init__(self, min_depth=1e-3, max_depth=10, depth_bins=64):
        super(DiscreteNLLLoss, self).__init__()
        self.name = 'CrossEntropy'
        self.ignore_index = -(depth_bins + 1)
        self._loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_bins = depth_bins
        self.alpha = 1
        self.zeta = 1 - min_depth
        self.beta = max_depth + self.zeta

    def quantize_depth(self, depth):
        depth = torch.log(depth / self.alpha) / np.log(self.beta / self.alpha)
        depth = depth * (self.depth_bins - 1)
        depth = torch.round(depth) 
        depth = depth.long()
        return depth
        

    def _dequantize_depth(self, depth):
        """
        Inverse of quantization
        depth : NCHW -> N1HW
        """
        depth = depth.float()
        depth = depth / (self.depth_bins - 1)  # Normalize to [0, 1]
        depth = depth * (np.log(self.beta / self.alpha))  # Scale to [0, log(beta/alpha)]
        depth = torch.exp(depth) * self.alpha  # Apply exponential scaling
        return depth

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        target = self.quantize_depth(target)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            mask = mask.long()
            input = input * mask + (1 - mask) * self.ignore_index
            target = target * mask + (1 - mask) * self.ignore_index

        input = input.permute(0, 2, 3, 1)
        input = input.flatten(start_dim=0, end_dim=-2)
        target = target.flatten(start_dim=0, end_dim=-2).squeeze(-1)

        loss = self._loss_func(input, target)

        if not return_interpolated:
            return loss

        return loss, intr_input

# ----------------- START OF NEW CODE -----------------

class CombinedLoss(nn.Module):
    def __init__(self, beta=0.15):
        super(CombinedLoss, self).__init__()
        self.silog_loss = SILogLoss(beta=beta)
        self.rmse_weight = 0.5
        self.berhu_loss = BerHuLoss()
        self.msgrad_loss = MSGradientLoss()
        self.photometric_loss = PhotometricLoss()

    def rmse_loss(self, pred, target):
        return torch.sqrt(torch.mean((pred - target) ** 2))

    def forward(self, input, target, mask=None):
        silog = self.silog_loss(input, target, mask)
        rmse = self.rmse_loss(input, target)
        berhu = self.berhu_loss(input, target)
        msgrad = self.msgrad_loss(input, target, mask)
        photometric = self.photometric_loss(input, target)
        
        combined_loss = silog + self.rmse_weight * rmse + berhu + msgrad + photometric
        
        return combined_loss

# ----------------- END OF NEW CODE -----------------

