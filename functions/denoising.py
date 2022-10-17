import torch
import torch.nn.functional as F
from tool.utils import RequiresGradContext
import torch.autograd as autograd


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out

def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)

def cosine_similarity(X,Y):
    '''
    compute cosine similarity for each pair of image
    Input shape: (batch,channel,H,W)
    Output shape: (batch,1)
    '''
    b, c, h, w = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    X = X.reshape(b, c, h * w)
    Y = Y.reshape(b, c, h * w)
    corr = norm(X)*norm(Y)#(B,C,H*W)
    similarity = corr.sum(dim=1).mean(dim=1)
    return similarity


def mse(x,y):
    return (x - y).square().sum(dim=(1, 2, 3))

def egsde_sample(y, dse,ls,die,li,t,model,logvar,betas,xt,s1,s2, model_name = 'ddim'):
    # mean for SDE
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    with torch.no_grad():
        if model_name == 'DDPM':
            model_output = model(y, t)
        elif model_name == 'ADM':
            model_output = model(y, t)
            model_output, _ = torch.split(model_output, 3, dim=1)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, y.shape) * (y - extract(weighted_score, t, y.shape) * model_output)

    # energy-guidance
    weight_energy = betas / torch.sqrt(alphas)
    weight_t = extract(weight_energy, t, y.shape)
    ## realistic expert2 based on dse
    with RequiresGradContext(y, requires_grad=True):
        Y = dse(y,t)
        X = dse(xt,t)
        if s1 == 'cosine':
            energy = cosine_similarity(Y,X)
        if s1 == 'neg_l2':
            energy = - mse(Y,X)
        grad = autograd.grad(energy.sum(), y)[0]
    mean = mean - ls * weight_t * grad.detach()
    ## faithful expert based on die
    down, up = die
    with RequiresGradContext(y, requires_grad=True):
        Y = up(down(y))
        X = up(down(xt))
        if s2 == 'cosine':
            energy = - cosine_similarity(X, Y)
        if s2 == 'neg_l2':
            energy =  mse(X, Y)
        grad = autograd.grad(energy.sum(), y)[0]
    mean = mean - li * weight_t * grad.detach()

    #add noise
    logvar = extract(logvar, t, y.shape)
    noise = torch.randn_like(y)
    mask = 1 - (t == 0).float()
    mask = mask.reshape((y.shape[0],) + (1,) * (len(y.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample




