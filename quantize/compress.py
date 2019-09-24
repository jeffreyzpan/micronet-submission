import torch
import numpy as np
from .ttq import calc_threshold
from sklearn.cluster import KMeans
from torch.nn.parameter import Parameter

def _to_nbits(qweight, n, format_only=False):

    k = 8 // n
    # calculate zipped length and pad to multiple of 4.
    wl = qweight.numel()
    zl = (wl - 1) // k + 1
    if wl < zl * k and not format_only:
        qweight.resize_(zl * k)
        qweight[wl:] = 0
    qweight = qweight.view(zl, k)

    # construct ByteTensor to hold zipped weights.
    zweight = torch.ByteTensor(zl, 1)

    if not format_only:
        torch.mm(qweight, torch.ByteTensor([[2 ** (n*i)] for i in range(k-1, -1, -1)]),
                 out=zweight)

    return zweight


def _from_nbits(zweight, n):

    k = 8 // n
    zl = zweight.numel()
    qweight = zweight.new(zl, k)

    for i in range(k):
        div = 2 ** (n*(k-1-i))
        qweight[:, i] = zweight / div
        zweight %= div

    return qweight


def _zip_ttq(ctx, format_only=False):

    if not format_only:
        # first quantize full precision laten weight.
        fp_weight = ctx.weight.data.cpu()
        threshold = calc_threshold(fp_weight.abs())
        mask_p = fp_weight.new()
        mask_n = fp_weight.new()
        mask_z = fp_weight.new()
        torch.gt(fp_weight, threshold, out=mask_p)
        torch.lt(fp_weight, -threshold, out=mask_n)
        torch.add(mask_p, mask_n, out=mask_z)
        scale = torch.dot(fp_weight.abs().view(-1), mask_z.view(-1)) / mask_z.sum()
        ctx.wp.data *= scale
        ctx.wn.data *= scale

        qweight = fp_weight.byte().fill_(1)
        qweight.masked_fill_(mask_p.byte(), 2)
        qweight.masked_fill_(mask_n.byte(), 0)
    else:
        qweight = ctx.weight.data.byte()

    ctx.weight.data = _to_nbits(qweight, 2, format_only)


def _unzip_ttq(ctx):

    qweight = _from_nbits(ctx.weight.data, 2)
    ctx.weight.data = qweight.resize_(ctx.weight_size).float() - 1


def _zip_kmeans(ctx, format_only=False):

    if not format_only:
        k = 2 ** ctx.nbits
        size = ctx.weight_size
        weight_numpy = ctx.weight.data.view(-1, 1).cpu().numpy()
        init = np.linspace(np.min(weight_numpy), np.max(weight_numpy), k)
        init = init.reshape(k, 1)
        codebook = KMeans(n_clusters=k, init=init, n_init=1, n_jobs=-1).fit(weight_numpy)
        ctx.weight_centers = Parameter(torch.from_numpy(codebook.cluster_centers_).float())
        qweight = torch.from_numpy(codebook.labels_).byte()
    else:
        qweight = ctx.weight.data.byte()
        ctx.weight_centers = Parameter(torch.FloatTensor(2**ctx.nbits, 1))

    ctx.weight.data = _to_nbits(qweight, ctx.nbits, format_only)


def _unzip_kmeans(ctx, format_only=False):

    qweight = _from_nbits(ctx.weight.data, ctx.nbits).view(-1)
    ctx.weight.data = ctx.weight_centers.data[qweight.long()]\
        .resize_(ctx.weight_size).contiguous()


def zip(ctx, format_only=False):
    ctx.weight_size = ctx.weight.data.size()
    if ctx.method == 'TTQ':
        _zip_ttq(ctx, format_only)
    elif ctx.method == 'KMeans':
        _zip_kmeans(ctx, format_only)


def unzip(ctx):
    if ctx.method == 'TTQ':
        _unzip_ttq(ctx)
    elif ctx.method == 'KMeans':
        _unzip_kmeans(ctx)
