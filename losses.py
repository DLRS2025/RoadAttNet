import torch
import torch.nn.functional as F

EPS = 1e-5

def dice_loss(y_true, y_pred):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    inter = torch.sum(y_true_f * y_pred_f)
    denom = torch.sum(y_true_f) + torch.sum(y_pred_f)
    return 1.0 - (2.0 * inter + EPS) / (denom + EPS)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_pred = torch.clamp(y_pred, EPS, 1.0 - EPS)
    p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
    loss = -alpha_t * torch.pow(1.0 - p_t, gamma) * torch.log(p_t)
    return torch.mean(loss)

def sobel_edges(img):
    B, C, H, W = img.shape
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=img.dtype, device=img.device)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=img.dtype, device=img.device)
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    
    grad_x = F.conv2d(img, sobel_x, padding=1, groups=C)
    grad_y = F.conv2d(img, sobel_y, padding=1, groups=C)
    return grad_x, grad_y

def boundary_aware_loss(y_true, y_pred, w=5.0):
    y_pred = torch.clamp(y_pred, EPS, 1.0 - EPS)
    grad_x, grad_y = sobel_edges(y_true)
    mag = torch.sqrt(grad_x**2 + grad_y**2 + EPS)
    e = torch.clamp(mag / 4.0, 0.0, 1.0)
    weights = 1.0 + w * e
    bce = -(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))
    return torch.mean(weights * bce)

def composite_loss(y_true, y_pred, lam_d=0.4, lam_f=0.4, lam_b=0.2):
    ld = dice_loss(y_true, y_pred)
    lf = focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0)
    lb = boundary_aware_loss(y_true, y_pred, w=5.0)
    return lam_d * ld + lam_f * lf + lam_b * lb
