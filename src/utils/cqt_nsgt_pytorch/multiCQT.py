
import torch
import torch.nn as nn
from utils.cqt_nsgt_pytorch.CQT_nsgt import CQT_nsgt


class multi_CQT(nn.Module):
    """
    Combines CQTs with different resolutions for different octaves.
    """
    def __init__(self,
                numocts=int,
                binsoct=list,
                num_time_frames= None,
                fmax=list,
                mode: str = "oct",
                window: str = "hann",
                fs: int = 44100,
                audio_len: int = 44100,
                dtype = torch.float32,
                device: str = "cpu",
                ):
        """
            args:
                numocts (int) number of octaves
                binsoct (list) numbers of bins per octave. Can be a list of lists if mode="flex_oct"
                mode (string) defines the mode of operation:
                     "critical": (default) critical sampling (no redundancy) returns a list of tensors, each with different time resolution (slow implementation)
                     "critical_fast": notimplemented
                     "matrix": returns a 2d-matrix maximum redundancy (discards DC and Nyquist)
                     "matrix_pow2": returns a 2d-matrix maximum redundancy (discards DC and Nyquist) (time-resolution is rounded up to a power of 2)
                     "matrix_complete": returns a 2d-matrix maximum redundancy (with DC and Nyquist)
                     "matrix_slow": returns a 2d-matrix maximum redundancy (slow implementation)
                     "oct": octave-wise rasterization ( modearate redundancy) returns a list of tensors, each from a different octave with different time resolution (discards DC and Nyquist)
                     "oct_complete": octave-wise rasterization ( modearate redundancy) returns a list of tensors, each from a different octave with different time resolution (with DC and Nyquist)
                fs (float) sampling frequency
                audio_len (int) sample length
                device
        """
        super().__init__()

        # CQT parameters
        self.num_octs = numocts
        self.bins_per_oct = [int(b) for b in binsoct]
        self.num_time_frames = num_time_frames
        self.fmax = fmax
        self.mode = mode
        self.window = window
        self.fs = fs
        self.audio_len = audio_len
        self.dtype = dtype
        self.device = device

        if self.window == "kaiser":
            self.win = ("kaiser", 1)
        else:
            self.win = self.window
        
        self.frqs = []
        self.frqs_comb = []
        cqts = []

        for cqt_index in range(len(self.bins_per_oct)):
            if self.fmax[cqt_index] < self.fs/2:

                cqt = CQT_nsgt(
                            numocts=self.num_octs[cqt_index] + 1, 
                            binsoct=self.bins_per_oct[cqt_index],
                            M=self.num_time_frames,
                            fmax=self.fmax[cqt_index] * 2,
                            mode=self.mode,
                            window=self.win,
                            fs=self.fs,
                            audio_len=self.audio_len,
                            dtype=torch.float32,
                            device=self.device
                            )
                
            else:

                cqt = CQT_nsgt(
                            numocts=self.num_octs[cqt_index], 
                            binsoct=self.bins_per_oct[cqt_index],
                            M=self.num_time_frames,
                            fmax=self.fmax[cqt_index],
                            mode=self.mode,
                            window=self.win,
                            fs=self.fs,
                            audio_len=self.audio_len,
                            dtype=torch.float32,
                            device=self.device
                            )                


            self.frqs.append(cqt.frqs)
            cqts.append(cqt)

        self.cqts = cqts

    def apply_hpf_DC(self, x):
        #

        Hlpf=torch.zeros(self.cqts[0].Ls, dtype=self.dtype, device=self.device)

        Hlpf[0:len(self.cqts[0].g[0])//2]=self.cqts[0].g[0][:len(self.cqts[0].g[0])//2]*self.cqts[0].gd[0][:len(self.cqts[0].g[0])//2]*self.cqts[0].M[0]
        Hlpf[-len(self.cqts[0].g[0])//2:]=self.cqts[0].g[0][len(self.cqts[0].g[0])//2:]*self.cqts[0].gd[0][len(self.cqts[0].g[0])//2:]*self.cqts[0].M[0]
        #filter nyquist
        nyquist_idx=len(self.cqts[-1].g)//2
        Lg=len(self.cqts[-1].g[nyquist_idx])
        Hlpf[self.cqts[-1].wins[nyquist_idx][0:(Lg+1)//2]]+=self.cqts[-1].g[nyquist_idx][(Lg)//2:]*self.cqts[-1].gd[nyquist_idx][(Lg)//2:]*self.cqts[-1].M[nyquist_idx]
        Hlpf[self.cqts[-1].wins[nyquist_idx][-(Lg-1)//2:]]+=self.cqts[-1].g[nyquist_idx][:(Lg)//2]*self.cqts[-1].gd[nyquist_idx][:(Lg)//2]*self.cqts[-1].M[nyquist_idx]

        Hhpf=1-Hlpf

        #return self.cqts[-1].apply_hpf_DC(x)
        Lin=x.shape[-1]
        if Lin<self.cqts[0].Ls:
            #pad zeros
            x=torch.nn.functional.pad(x, (0, self.cqts[0].Ls-Lin))
        elif Lin> self.cqts[0].Ls:
            raise ValueError("Input signal is longer than the maximum length. I could have patched it, but I didn't. sorry :(")

        X=torch.fft.fft(x)
        X=X*torch.conj(Hhpf)
        out= torch.fft.ifft(X).real
        if Lin<self.cqts[0].Ls:
            out=out[..., :Lin]
        return out

    def fwd(self, x):
        """
        args:   
            x: input audio tensor
        output: 
            cqts: list of CQTs
        """

        # Compute multiple CQTs
        cqts = []

        self.shapes_fwd=[]

        for cqt_index in range(len(self.cqts)):

            cqt_aux = self.cqts[cqt_index].fwd(x)

            # # Cancel Nyquist component if fmax < fs/2
            # if self.fmax[cqt_index] < self.fs/2:
            #     cqt_aux[-1][:,:,:,:] = 0

            cqts.append(cqt_aux)

            shape_list=[]
            for i in range(len(cqt_aux)):
                shape_list.append(cqt_aux[i].shape)
            self.shapes_fwd.append(shape_list)
        

        cqt_list=[]


        for i, c in enumerate(cqts):
            if i==len(cqts)-1:
                for j, c2 in enumerate(c[0:len(c)]):
                    cqt_list.append(c2)
            else:
                for j, c2 in enumerate(c[0:len(c)-1]):
                    cqt_list.append(c2)
                

        return cqt_list
    
    def bwd(self, X_in):
        """"
        Inverse CQT for multiple resolutions.
        
        Args:
            X (list): List of CQTs.
        Returns:
            x (list): time-domain signals.
        """
        
        x = []

        cqt_1=[X_in[0], X_in[1], X_in[2], torch.zeros(self.shapes_fwd[0][3]).to(X_in[0].device).to(X_in[0].dtype)]
        cqt_2=[X_in[3], X_in[4], X_in[5], X_in[6], torch.zeros(self.shapes_fwd[1][4]).to(X_in[0].device).to(X_in[0].dtype)]
        cqt_3=[X_in[7], X_in[8]]
        X = [cqt_1, cqt_2, cqt_3]
        #print(self.shapes_fwd)

        # Perform the inverse CQT
        for cqt_index in range(len(self.cqts)):
            # Apply the inverse CQT to each CQT in the list
            x.append(self.cqts[cqt_index].bwd(X[cqt_index]))

        x_out = torch.zeros_like(x[0])
        for i in range(len(x)):
            # x_out += x[i]*(self.bins_per_oct[-1]/self.bins_per_oct[i])**2
            x_out += x[i]

        return x_out
