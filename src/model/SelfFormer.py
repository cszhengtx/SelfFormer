import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling
import torch
from typing import Tuple
import random

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dilation, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = ShiftConv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv2 = ShiftConv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=dilation,
                               stride=1, groups=dw_channel, dilation=dilation)
        self.conv3 = ShiftConv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ShiftConv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = ShiftConv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv5 = ShiftConv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, is_shift=True):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x, is_shift)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        if not is_shift:
            x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        if not is_shift:
            x = self.dropout2(x)

        z = y + x * self.gamma

        return z


def rotate(x, angle):
    """Rotate images by 90 degrees clockwise. Can handle any 2D data format.
    Args:
        x (Tensor): Image or batch of images.
        angle (int): Clockwise rotation angle in multiples of 90.
        data_format (str, optional): Format of input image data, e.g. BCHW,
            HWC. Defaults to BCHW.
    Returns:
        Tensor: Copy of tensor with rotation applied.
    """
    h_dim, w_dim = 2, 3

    if angle == 0:
        return x
    elif angle == 90:
        return x.flip(w_dim).transpose(h_dim, w_dim)
    elif angle == 180:
        return x.flip(w_dim).flip(h_dim)
    elif angle == 270:
        return x.flip(h_dim).transpose(h_dim, w_dim)
    else:
        raise NotImplementedError("Must be rotation divisible by 90 degrees")

class Crop2d(nn.Module):
    """Crop input using slicing. Assumes BCHW data.

    Args:
        crop (Tuple[int, int, int, int]): Amounts to crop from each side of the image.
            Tuple is treated as [left, right, top, bottom]/
    """

    def __init__(self, crop: Tuple[int, int, int, int]):
        super().__init__()
        self.crop = crop
        assert len(crop) == 4

    def forward(self, x):
        (left, right, top, bottom) = self.crop
        x0, x1 = left, x.shape[-1] - right
        y0, y1 = top, x.shape[-2] - bottom
        return x[:, :, y0:y1, x0:x1]


class Shift2d(nn.Module):
    """Shift an image in either or both of the vertical and horizontal axis by first
    zero padding on the opposite side that the image is shifting towards before
    cropping the side being shifted towards.

    Args:
        shift (Tuple[int, int]): Tuple of vertical and horizontal shift. Positive values
            shift towards right and bottom, negative values shift towards left and top.
    """

    def __init__(self, shift: Tuple[int, int]):
        super().__init__()
        self.shift = shift
        vert, horz = self.shift
        y_a, y_b = abs(vert), 0
        x_a, x_b = abs(horz), 0
        if vert < 0:
            y_a, y_b = y_b, y_a
        if horz < 0:
            x_a, x_b = x_b, x_a
        # Order : Left, Right, Top Bottom
        self.pad = nn.ZeroPad2d((x_a, x_b, y_a, y_b))
        self.crop = Crop2d((x_b, x_a, y_b, y_a))
        self.shift_block = nn.Sequential(self.pad, self.crop)

    def forward(self, x):
        return self.shift_block(x)

class ShiftConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """Custom convolution layer as defined by Laine et al. for restricting the
        receptive field of a convolution layer to only be upwards. For a h × w kernel,
        a downwards offset of k = [h/2] pixels is used. This is applied as a k sized pad
        to the top of the input before applying the convolution. The bottom k rows are
        cropped out for output.
        """
        super().__init__(*args, **kwargs)
        self.shift_size = (self.kernel_size[0] // 2 + (self.dilation[0]//2)*2, 0)
        # Use individual layers of shift for wrapping conv with shift
        shift = Shift2d(self.shift_size)
        self.pad = shift.pad
        self.crop = shift.crop

    def forward(self, x, is_shift=True):
        if is_shift:
            x = self.pad(x)
            x = super().forward(x)
            x = self.crop(x)
        else:
            x = super().forward(x)
        return x



class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        # self.body = nn.Sequential(ShiftConv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
        #                           nn.PixelUnshuffle(2))
        self.body = nn.Sequential(ShiftConv2d(channels, channels*2, kernel_size=3, stride=2, padding=1),
                              nn.ReLU())

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                              ShiftConv2d(channels, channels // 2, 1))
    def forward(self, x):
        return self.body(x)


class Intro(nn.Module):
    def __init__(self, in_channels, width, dilation = 2):
        super(Intro, self).__init__()
        self.conv1 = ShiftConv2d(in_channels, width, 3, stride=1, padding=dilation, dilation=dilation)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = ShiftConv2d(width, width, 3, stride=1, padding=dilation, dilation=dilation)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, is_shift=True):
        x = self.conv1(x, is_shift)
        x = self.relu1(x)
        x = self.conv2(x, is_shift)
        x = self.relu2(x)
        return x

class Down(nn.Module):
    def __init__(self, chan):
        super(Down, self).__init__()
        self.conv = nn.Conv2d(chan, chan*2, 1)
        self.shift = Shift2d((1, 0))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, is_shift=True):
        x = self.conv(x)
        x = self.shift(x)
        x = self.pool(x)
        return x

class Up(nn.Module):
    def __init__(self, chan):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = ShiftConv2d(chan, chan, 1)
        self.conv2 = ShiftConv2d(chan, chan // 2, 1)

    def forward(self, x, is_shift=True):
        x = self.up(x)
        x = self.conv1(x, is_shift)
        x = self.conv2(x, is_shift)
        return x

class GridSANaive(nn.Module):
    def __init__(self, embed_size, f_scale, stride):
        super().__init__()
        in_ch = embed_size
        self.conv_1 = ShiftConv2d(in_ch, in_ch//4, kernel_size=1)
        self.conv1_act = nn.ReLU(inplace=True)
        self.conv_2 = ShiftConv2d(in_ch//4, in_ch//4, kernel_size=3, stride=1, padding=stride, dilation=stride)
        self.conv2_act = nn.ReLU(inplace=True)
        self.conv_3 = ShiftConv2d(in_ch//4, in_ch, kernel_size=1)

    def _get_ff(self, x):
        x = self.conv1_act(self.conv_1(x))

        x = self.conv2_act(self.conv_2(x))

        x = self.conv_3(x)

        return x

    def forward(self, x):
        return x + self._get_ff(x)


class GridAttention(GridSANaive):
    def __init__(self, embed_size, f_scale, stride):
        super(GridAttention, self).__init__(embed_size, f_scale, stride)
        self.embed_size = embed_size // f_scale
        self.f_scale = f_scale
        self.stride = stride
        scale = 4**2  # for speed up if feel it slow else f_scale *= 1 for performance
        self.wqk = nn.Parameter(torch.randn(embed_size // scale, embed_size // scale))
        self.conv1 = nn.Sequential(nn.Conv2d(embed_size, embed_size // scale, kernel_size=1, stride=1, padding=0),
                                   nn.Conv2d(embed_size // scale, embed_size // scale, kernel_size=1, stride=1, padding=0))
        self.conv2 = nn.Conv2d(embed_size // scale, embed_size, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(embed_size, embed_size, kernel_size=1, padding=0, stride=1, groups=1)

    def _pad_for_shuffle(self, x, f):
        _, _, h, w = x.size()
        ph, pw = (f - h % f) % f, (f - w % f) % f
        x = F.pad(x, (0, pw, 0, ph))
        return x, ph, pw

    def _pixel_unshuffle(self, x, f):
        b, c, h, w = x.shape
        x = x.view(b, c, h // f, f, w // f, f)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c * f * f, h // f, w // f)

    def _pixel_shuffle(self, x, f):
        b, c, h, w = x.shape
        x = x.view(b, f, f, c // (f * f), h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(b, c // (f * f), h * f, w * f)

    def _get_attention(self, x, f, is_shift=True):
        _, c, h, w = x.shape

        if is_shift:
            shift = Shift2d((h // 2, 0))  # The size of the shift can be adjusted as needed
            x = shift.pad(x)

        xx = F.layer_norm(x, x.shape[-3:])
        xx, ph, pw = self._pad_for_shuffle(xx, f)
        xx = self._pixel_unshuffle(xx, f)
        xx = rearrange(xx, 'b (f c) h w -> (b f) c h w', c=c, f=f ** 2)

        # v, _, _ = self._pad_for_shuffle(x, f)
        # v = self._pixel_unshuffle(v, f)
        # v = rearrange(v, 'b (f c) h w -> (b f) c h w', c=c, f=f ** 2)
        v = xx

        b, _, sh, sw = xx.shape

        qk = rearrange(xx, 'b c h w -> (b h w) c')
        v = rearrange(v, 'b c h w -> (b h w) c')
        qk = torch.mm(qk, self.wqk)

        qk = rearrange(qk, '(b h w) k -> b (h w) k', b=b, h=sh, w=sw)
        v = rearrange(v, '(b h w) k -> b (h w) k', b=b, h=sh, w=sw)

        qk_norm = torch.linalg.norm(qk, dim=-1, keepdim=True) + 1e-8
        qk = qk / qk_norm

        attn = torch.bmm(qk, qk.transpose(1, 2))
        attn = (attn + 1.) / self.embed_size ** 0.5
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)

        out = rearrange(out, 'b (h w) e -> b e h w', b=b, h=sh, w=sw)
        out = rearrange(out, '(b f) c h w -> b (f c) h w', f=f ** 2, c=c)
        out = self._pixel_shuffle(out, f)

        if ph > 0:
            out = out[:, :, :-ph, :]
        if pw > 0:
            out = out[:, :, :, :-pw]

        if is_shift:
            out = shift.crop(out)
        return out

    def forward(self, x, is_shift=True):
        f = self.f_scale * self.stride

        return x + self._get_ff(x + self.conv(self.conv2(self._get_attention(self.conv1(x), f, is_shift))))


class FFN(nn.Module):
    def __init__(self, in_ch, FFN_Expand=4, gamma=1.):
        super(FFN, self).__init__()
        ffn_channel = FFN_Expand * in_ch
        self.conv1 = nn.Conv2d(in_ch, ffn_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(ffn_channel//2, in_ch, kernel_size=1)
        self.sg = SimpleGate()
        self.norm = LayerNorm2d(in_ch)
        self.gamma = gamma

    def forward(self, x):
        y = x
        x = self.conv1(self.norm(y))
        x = self.sg(x)
        x = self.conv2(x)

        z = y + x * self.gamma
        return z


class SelfFormer_block(nn.Module):
    def __init__(self, chan, dilation, f_scale):
        super(SelfFormer_block, self).__init__()
        self.dsca = NAFBlock(chan, dilation)
        self.dsa = GridAttention(embed_size=chan, f_scale=f_scale, stride=2)

    def forward(self, x, is_shift=True):
        x = self.dsca(x, is_shift)
        x = self.dsa(x, is_shift)
        return x


class SelfFormer_Seq(nn.Module):
    def __init__(self, chan, dilation, num, f_scale):
        super(SelfFormer_Seq, self).__init__()
        self.encoder = nn.ModuleList()
        for i in range(num):
            self.encoder.append(SelfFormer_block(chan, dilation, f_scale))

    def forward(self, x, is_shift=True):
        for encoder in self.encoder:
            x = encoder(x, is_shift)
        return x


class SelfFormer(nn.Module):
    def __init__(self, blindspot = 7, in_ch=3, out_ch=3, width=48, dilation=1,
                 enc_blk_nums=[3,4], middle_blk_nums=4, dec_blk_nums=[4,3]):
        super().__init__()
        self.dilation = dilation
        self.blindspot = blindspot
        in_channels = in_ch
        out_channels = out_ch

        self.intro = Intro(in_channels, width)
        self.output_conv = nn.Conv2d(width, out_channels, 1)
        self.output_block = nn.Sequential(
            nn.Conv2d(4 * width, 4 * width, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(4 * width,  width, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Conv2d(width, width, 1),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.output_conv,
        )
        self.output_sca = NAFBlock(width, 2)
        self.output_block_2 = nn.Sequential(
            nn.Conv2d(4 * width, 4 * width, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(4 * width, width, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Conv2d(width, width, 1),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.output_sca,

            nn.Conv2d(width, out_channels, 1),
        )   # change to use apbsn's pd maybe faster while reaching good performance

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        f_scale = 8
        for num in enc_blk_nums:
            self.encoders.append(
                SelfFormer_Seq(chan, dilation, num, f_scale)
            )
            self.downs.append(
                Down(chan)
            )

            chan = chan * 2
            f_scale = f_scale//2

        self.middle_blks = SelfFormer_Seq(chan, dilation, middle_blk_nums, f_scale)

        for num in dec_blk_nums:
            self.ups.append(
                Up(chan)
            )
            chan = chan // 2
            f_scale = f_scale*2
            self.decoders.append(
                SelfFormer_Seq(chan, dilation, num, f_scale)
            )

        self.pad = nn.ZeroPad2d((0, 0, 2, 0))
        self.crop = Crop2d((0, 0, 0, 2))

    def forward(self, x, shift=None, refine=False, is_shift=True):
        if shift is not None:
            self.shift = Shift2d(((self.blindspot + 1) // 2, 0))
        else:
            self.shift = Shift2d(((self.blindspot + 1) // 2, 0))
        if refine:
            self.shift = Shift2d(((self.blindspot + 1) // 2, 0))

        rotated = [rotate(x, rot) for rot in (0, 90, 180, 270)]
        x = torch.cat((rotated), dim=0)

        x = self.intro(x, is_shift)

        encs = []
        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            x = encoder(x, is_shift)
            encs.append(x)
            x = down(x, is_shift)

        x = self.middle_blks(x, is_shift)

        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x, is_shift)
            x = x + enc_skip
            x = decoder(x, is_shift)

        # Apply shift
        if is_shift:
            shifted = self.shift(x)
        else:
            shifted = x

        # Unstack, rotate and combine
        rotated_batch = torch.chunk(shifted, 4, dim=0)
        aligned = [
            rotate(rotated, rot)
            for rotated, rot in zip(rotated_batch, (0, 270, 180, 90))
        ]
        x = torch.cat(aligned, dim=1)

        if is_shift:
            x = self.output_block(x)
        else:
            x = self.output_block_2(x)

        return x