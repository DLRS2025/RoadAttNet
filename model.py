import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import composite_loss

def _bilinear_sample(img, coords_y, coords_x):
    # img: [B, C, H, W]
    # coords_y, coords_x: [B, 1, H, W]
    B, C, H, W = img.shape
    
    # Normalize coordinates to [-1, 1] for grid_sample
    norm_x = 2.0 * coords_x / (W - 1) - 1.0
    norm_y = 2.0 * coords_y / (H - 1) - 1.0
    
    # grid_sample expects grid of shape [B, H, W, 2] with (x, y) coordinates
    grid = torch.cat([norm_x, norm_y], dim=1) # [B, 2, H, W]
    grid = grid.permute(0, 2, 3, 1) # [B, H, W, 2]
    
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)


class OrientedCoordinateAttention(nn.Module):
    def __init__(self, in_channels, length=9, reduction=8):
        super().__init__()
        self.length = int(length)
        self.reduction = int(reduction)
        hidden = max(8, in_channels // self.reduction)
        
        self.theta_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.theta_conv1 = nn.Sequential(
            nn.Conv2d(hidden, 1, 1, padding=0),
            nn.Sigmoid()
        )
        self.attn_reduce = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden, 1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.attn_expand = nn.Sequential(
            nn.Conv2d(hidden, 2 * in_channels, 1, padding=0),
            nn.Sigmoid()
        )

    def _oriented_pool(self, x, vx, vy):
        B, C, H, W = x.shape
        device = x.device
        
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device), 
                                torch.arange(W, dtype=torch.float32, device=device), indexing="ij")
        yy = yy.reshape(1, 1, H, W).expand(B, -1, -1, -1)
        xx = xx.reshape(1, 1, H, W).expand(B, -1, -1, -1)
        
        half = self.length // 2
        offsets = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
        
        acc = 0.0
        for t in offsets:
            coords_y = yy + t * vy
            coords_x = xx + t * vx
            sample = _bilinear_sample(x, coords_y, coords_x)
            acc = acc + sample
        return acc / float(self.length)

    def forward(self, x):
        theta = np.pi * self.theta_conv1(self.theta_conv3(x))
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        vtan_x, vtan_y = cos_t, sin_t
        vnor_x, vnor_y = -sin_t, cos_t
        
        tan_feat = self._oriented_pool(x, vtan_x, vtan_y)
        nor_feat = self._oriented_pool(x, vnor_x, vnor_y)
        
        context = torch.cat([tan_feat, nor_feat], dim=1)
        w = self.attn_expand(self.attn_reduce(context))
        
        alpha_tan, alpha_norm = torch.chunk(w, 2, dim=1)
        return (alpha_tan + alpha_norm) * x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        
        if in_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, 1, padding=0, bias=False),
                nn.BatchNorm2d(filters)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + shortcut
        x = self.relu2(x)
        return x


class MultiscaleRGBBranch(nn.Module):
    def __init__(self, in_channels, base_filters=32):
        super().__init__()
        self.f0 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.pool2 = nn.AvgPool2d(2)
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.pool4 = nn.AvgPool2d(4)
        self.conv_d4 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.ms_conv = nn.Sequential(
            nn.Conv2d(base_filters * 3, base_filters * 2, 1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb):
        f0 = self.f0(rgb)
        
        d2 = self.pool2(rgb)
        d2 = self.conv_d2(d2)
        u2 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        
        d4 = self.pool4(rgb)
        d4 = self.conv_d4(d4)
        u4 = F.interpolate(d4, scale_factor=4, mode="bilinear", align_corners=False)
        
        ms = torch.cat([f0, u2, u4], dim=1)
        ms = self.ms_conv(ms)
        return ms


class MultidimPriorBranch(nn.Module):
    def __init__(self, in_channels, out_filters=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_filters, out_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_filters, out_filters, 1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, prior):
        return self.net(prior)


class RoadAttNetCore(nn.Module):
    def __init__(self, base_filters=64, oca_length=9):
        super().__init__()
        
        self.rgb_ms = MultiscaleRGBBranch(3, base_filters=base_filters // 2)
        self.prior_f = MultidimPriorBranch(1, out_filters=base_filters)
        
        in_ch = (base_filters // 2) * 2 + base_filters
        
        self.e1 = ResidualBlock(in_ch, base_filters)
        self.p1 = nn.MaxPool2d(2)
        
        self.e2 = ResidualBlock(base_filters, base_filters * 2)
        self.p2 = nn.MaxPool2d(2)
        
        self.e3 = ResidualBlock(base_filters * 2, base_filters * 4)
        self.p3 = nn.MaxPool2d(2)
        
        self.e4 = ResidualBlock(base_filters * 4, base_filters * 8)
        self.p4 = nn.MaxPool2d(2)
        
        self.bott = ResidualBlock(base_filters * 8, base_filters * 16)
        
        self.oca1 = OrientedCoordinateAttention(base_filters * 16 + base_filters * 8, length=oca_length)
        self.d4_res = ResidualBlock(base_filters * 16 + base_filters * 8, base_filters * 8)
        self.aux1 = nn.Sequential(nn.Conv2d(base_filters * 8, 1, 1), nn.Sigmoid())
        
        self.oca2 = OrientedCoordinateAttention(base_filters * 8 + base_filters * 4, length=oca_length)
        self.d3_res = ResidualBlock(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.aux2 = nn.Sequential(nn.Conv2d(base_filters * 4, 1, 1), nn.Sigmoid())
        
        self.oca3 = OrientedCoordinateAttention(base_filters * 4 + base_filters * 2, length=oca_length)
        self.d2_res = ResidualBlock(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.aux3 = nn.Sequential(nn.Conv2d(base_filters * 2, 1, 1), nn.Sigmoid())
        
        self.oca4 = OrientedCoordinateAttention(base_filters * 2 + base_filters, length=oca_length)
        self.d1_res = ResidualBlock(base_filters * 2 + base_filters, base_filters)
        self.main = nn.Sequential(nn.Conv2d(base_filters, 1, 1), nn.Sigmoid())

    def forward(self, x):
        rgb = x[:, :3, ...]
        prior = x[:, 3:4, ...]
        
        rgb_ms = self.rgb_ms(rgb)
        prior_f = self.prior_f(prior)
        
        x0 = torch.cat([rgb_ms, prior_f], dim=1)
        
        e1 = self.e1(x0)
        p1 = self.p1(e1)
        
        e2 = self.e2(p1)
        p2 = self.p2(e2)
        
        e3 = self.e3(p2)
        p3 = self.p3(e3)
        
        e4 = self.e4(p3)
        p4 = self.p4(e4)
        
        bott = self.bott(p4)
        
        d4 = F.interpolate(bott, scale_factor=2, mode="bilinear", align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.oca1(d4)
        d4 = self.d4_res(d4)
        aux1 = self.aux1(d4)
        aux1_up = F.interpolate(aux1, scale_factor=8, mode="bilinear", align_corners=False)
        
        d3 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.oca2(d3)
        d3 = self.d3_res(d3)
        aux2 = self.aux2(d3)
        aux2_up = F.interpolate(aux2, scale_factor=4, mode="bilinear", align_corners=False)
        
        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.oca3(d2)
        d2 = self.d2_res(d2)
        aux3 = self.aux3(d2)
        aux3_up = F.interpolate(aux3, scale_factor=2, mode="bilinear", align_corners=False)
        
        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.oca4(d1)
        d1 = self.d1_res(d1)
        main = self.main(d1)
        
        return main, aux1_up, aux2_up, aux3_up

def build_roadattnet_core(base_filters=64, oca_length=9):
    return RoadAttNetCore(base_filters=base_filters, oca_length=oca_length)

class RoadAttNet(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
        self.s1 = nn.Parameter(torch.zeros(1))
        self.s2 = nn.Parameter(torch.zeros(1))
        self.s3 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.core(x)

    def compute_loss(self, y_true, main, aux1, aux2, aux3):
        L_main = composite_loss(y_true, main)
        L1 = composite_loss(y_true, aux1)
        L2 = composite_loss(y_true, aux2)
        L3 = composite_loss(y_true, aux3)
        
        s1 = torch.clamp(self.s1, min=-5.0, max=5.0)
        s2 = torch.clamp(self.s2, min=-5.0, max=5.0)
        s3 = torch.clamp(self.s3, min=-5.0, max=5.0)
        
        aux_term = (
            0.5 * torch.exp(-2.0 * s1) * L1 + s1
            + 0.5 * torch.exp(-2.0 * s2) * L2 + s2
            + 0.5 * torch.exp(-2.0 * s3) * L3 + s3
        )
        
        loss = L_main + aux_term
        return loss, L_main, (L1 + L2 + L3) / 3.0
