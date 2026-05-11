import torch

def wasserstein_loss_D(real_scores, fake_scores):
    return -(real_scores.mean() - fake_scores.mean())

def wasserstein_loss_G(fake_scores):
    return -fake_scores.mean()

def gradient_penalty(D, real_imgs, fake_imgs, labels, device):
    B = real_imgs.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    
    scores = D(interpolated, labels)
    grad = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True
    )[0]
    
    grad = grad.view(B, -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp

