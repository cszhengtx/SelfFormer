import torch
import torch.nn as nn

from . import regist_model
from .SelfFormer import SelfFormer


@regist_model
class Denoiser(nn.Module):
    '''
    Asymmetric PD Blind-Spot Network (AP-BSN)
    '''
    def __init__(self, dilation, width,
                 enc_blk_nums, middle_blk_nums, dec_blk_nums,
                 R3=True, R3_T=8, R3_p=0.16, blindspot = 7):
        '''
        Args:
            dilation       : stride factor of bsn's dilated conv
            R3             : flag of 'Random Replacing Refinement'
            R3_T           : number of masks for R3
            R3_p           : probability of R3
            enc_blk_nums   : number of bsn encoder module
            middle_blk_nums: number of bsn mid-level module
            dec_blk_nums   : number of bsn decoder module
        '''
        super().__init__()

        # network hyper-parameters
        self.R3 = R3
        self.R3_T = R3_T
        self.R3_p = R3_p
        self.bsn = SelfFormer(blindspot=blindspot, in_ch=3, out_ch=3, width=width, dilation=dilation,
                                   enc_blk_nums=enc_blk_nums, middle_blk_nums=middle_blk_nums, dec_blk_nums=dec_blk_nums)

    def forward(self, x):
        return self.bsn(x), self.bsn(x, is_shift=False)
    
    def denoise(self, x):
        '''
        Denoising process for inference.
        '''

        img_pd_bsn = self.bsn(x, is_shift=False)

        # Random Replacing Refinement
        if not self.R3:
            ''' Directly return the result (w/o R3) '''
            return img_pd_bsn
        else:
            denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(x)
                mask = indice < self.R3_p

                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = x[mask]

                denoised[..., t] = self.bsn(tmp_input, refine=True, is_shift=False)

            return torch.mean(denoised, dim=-1)