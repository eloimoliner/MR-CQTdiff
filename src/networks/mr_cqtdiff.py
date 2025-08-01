import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math as m
import torch
from torch.utils.checkpoint import checkpoint
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

import einops

from utils.cqt_nsgt_pytorch.multiCQT import multi_CQT


def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel=(1,1), bias=False, dilation=1,
        init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.kernel=kernel
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel[0]*kernel[1], fan_out=out_channels*kernel[0]*kernel[1])
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel[0], kernel[1]], **init_kwargs) * init_weight) 
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        #f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding="same", dilation=self.dilation)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 1e-4, channel_last=True):
        """
        channel_last = False corresponds to (B, C, T) tensors
        channel_last = True corresponds to (T, B, C) tensors
        """
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x

class BiasFreeLayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-7):
        super(BiasFreeLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1,1,num_features))
        self.eps = eps

    def forward(self, x):
        N, T, C = x.size()

        std=x.std(-1, keepdim=True) #reduce over channels and time

        ## normalize
        x = (x) / (std+self.eps)

        return x * self.gamma


class BiasFreeGroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-7):
        super(BiasFreeGroupNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, F, T = x.shape

        # Convert symbolic integer to regular int if necessary
        self.num_groups=int(self.num_groups)
        gc = int(C // self.num_groups)
        
        # Rearrange dimensions
        x = einops.rearrange(x, 'n (g gc) f t -> n g (gc f t)', g=self.num_groups, gc=gc)

        # Compute standard deviation
        std = x.std(-1, keepdim=True)  # Reduce over channels and time

        # Normalize
        x = x / (std + self.eps)

        # Restore original shape
        x = einops.rearrange(x, 'n g (gc f t) -> n (g gc) f t', g=self.num_groups, gc=gc, f=F, t=T)

        return x * self.gamma



class RFF_MLP_Block(nn.Module):
    """
        Encoder of the noise level embedding
        Consists of:
            -Random Fourier Feature embedding
            -MLP
    """
    def __init__(self, emb_dim=512, rff_dim=32, init=None):
        super().__init__()
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, rff_dim]), requires_grad=False)
        self.MLP = nn.ModuleList([
            Linear(2*rff_dim, 128, **init),
            Linear(128, 256, **init),
            Linear(256, emb_dim, **init),
        ])

    def forward(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)

        Returns:
          x: embedding of sigma
              (shape: [B, 512], dtype: float32)
        """
        x = self._build_RFF_embedding(sigma)
        for layer in self.MLP:
            x = F.relu(layer(x))
        return x

    def _build_RFF_embedding(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)
        Returns:
          table:
              (shape: [B, 64], dtype: float32)
        """
        freqs = self.RFF_freq
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class AddFreqEncodingRFF(nn.Module):
    '''
    [B, T, F, 2] => [B, T, F, 12]  
    Generates frequency positional embeddings and concatenates them as 10 extra channels
    This function is optimized for F=1025
    '''
    def __init__(self, f_dim, N):
        super(AddFreqEncodingRFF, self).__init__()
        self.N=N
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, N]), requires_grad=False)


        self.f_dim=f_dim #f_dim is fixed
        embeddings=self.build_RFF_embedding()
        self.embeddings=nn.Parameter(embeddings, requires_grad=False) 

        
    def build_RFF_embedding(self):
        """
        Returns:
          table:
              (shape: [C,F], dtype: float32)
        """
        freqs = self.RFF_freq
        freqs=freqs.unsqueeze(-1) # [1, 32, 1]

        self.n=torch.arange(start=0,end=self.f_dim)
        self.n=self.n.unsqueeze(0).unsqueeze(0)  #[1,1,F]

        table = 2 * np.pi * self.n * freqs

        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1) #[1,32,F]

        return table
    

    def forward(self, input_tensor):

        batch_size_tensor = input_tensor.shape[0]  # get batch size
        time_dim = input_tensor.shape[-1]  # get time dimension

        fembeddings_2 = torch.broadcast_to(self.embeddings, [batch_size_tensor, time_dim,self.N*2, self.f_dim])
        fembeddings_2=fembeddings_2.permute(0,2,3,1)
    
        
        return torch.cat((input_tensor,fembeddings_2),1)  

        
class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        use_norm=True,
        num_dils = 6,
        bias=False,
        kernel_size=(5,3),
        emb_dim=512,
        proj_place='before', #using 'after' in the decoder out blocks
        init=None,
        init_zero=None,
        Fdim=128, #number of frequency bins
    ):
        super().__init__()

        self.bias=bias
        self.use_norm=use_norm
        self.num_dils=num_dils
        self.proj_place=proj_place
        self.Fdim=Fdim

        if self.proj_place=='before':
            #dim_out is the block dimension
            N=dim_out
        else:
            #dim in is the block dimension
            N=dim
            self.proj_out = Conv2d(N, dim_out,   bias=bias, **init) if N!=dim_out else nn.Identity() #linear projection

        self.res_conv = Conv2d(dim, dim_out, bias=bias, **init) if dim!= dim_out else nn.Identity() #linear projection
        self.proj_in = Conv2d(dim, N,   bias=bias, **init) if dim!=N else nn.Identity()#linear projection



        self.H=nn.ModuleList()
        self.affine=nn.ModuleList()
        self.gate=nn.ModuleList()
        if self.use_norm:
            self.norm=nn.ModuleList()

        for i in range(self.num_dils):

            if self.use_norm:
                self.norm.append(BiasFreeGroupNorm(N,8))

            self.affine.append(Linear(emb_dim, N, **init))
            self.gate.append(Linear(emb_dim, N, **init_zero))
            self.H.append(Conv2d(N,N,    
                                    kernel=kernel_size,
                                    dilation=(2**i,1),
                                    bias=bias, **init)) #freq convolution (dilated) 



    def forward(self, input_x, sigma, checkpointing=False):
        
        x=input_x

        #print class of self.proj_in
        x=self.proj_in(x)


        def conv_block(x, sigma):
            for norm, affine, gate, conv in zip(self.norm, self.affine, self.gate, self.H):
                x0 = x
                if self.use_norm:
                    x = norm(x)
    
                gamma = affine(sigma)
                scale = gate(sigma)
    
                x = x * (gamma.unsqueeze(2).unsqueeze(3) + 1)
    
                x = (x0 + conv(F.gelu(x)) * scale.unsqueeze(2).unsqueeze(3)) / (2**0.5)
    
            return x

        if checkpointing:
            x = checkpoint(conv_block, x, sigma, use_reentrant=False)
        else:
            x = conv_block( x, sigma)  # Use checkpointing here

        
        #one residual connection here after the dilated convolutions


        if self.proj_place=='after':
            x=self.proj_out(x)

        x=(x + self.res_conv(input_x))/(2**0.5)

        return x



_kernels = {
    'linear':
        [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    'cubic': 
        [-0.01171875, -0.03515625, 0.11328125, 0.43359375,
        0.43359375, 0.11328125, -0.03515625, -0.01171875],
    'lanczos3': 
        [0.003689131001010537, 0.015056144446134567, -0.03399861603975296,
        -0.066637322306633, 0.13550527393817902, 0.44638532400131226,
        0.44638532400131226, 0.13550527393817902, -0.066637322306633,
        -0.03399861603975296, 0.015056144446134567, 0.003689131001010537]
}
class UpDownResample(nn.Module):
    def __init__(self,
        up=False, 
        down=False,
        mode_resample="T", #T for time, F for freq, TF for both
        resample_filter='cubic', 
        pad_mode='reflect'
        ):
        super().__init__()
        assert not (up and down) #you cannot upsample and downsample at the same time
        assert up or down #you must upsample or downsample
        self.down=down
        self.up=up
        if up or down:
            #upsample block
            self.pad_mode = pad_mode #I think reflect is a goof choice for padding
            self.mode_resample=mode_resample
            if mode_resample=="T":
                kernel_1d = torch.tensor(_kernels[resample_filter], dtype=torch.float32)
            elif mode_resample=="F":
                #kerel shouuld be the same
                kernel_1d = torch.tensor(_kernels[resample_filter], dtype=torch.float32)
            else:
                raise NotImplementedError("Only time upsampling is implemented")
                #TODO implement freq upsampling and downsampling
            self.pad = kernel_1d.shape[0] // 2 - 1
            self.register_buffer('kernel', kernel_1d)
    def forward(self, x):
        shapeorig=x.shape
        x=x.view(-1,x.shape[-2],x.shape[-1]) #I have the feeling the reshape makes everything consume too much memory. There is no need to have the channel dimension different than 1. I leave it like this because otherwise it requires a contiguous() call, but I should check if the memory gain / speed, would be significant.
        if self.mode_resample=="F":
            x=x.permute(0,2,1)#call contiguous() here?

        if self.down:
            x = F.pad(x, (self.pad,) * 2, self.pad_mode)
        elif self.up:
            x = F.pad(x, ((self.pad + 1) // 2,) * 2, self.pad_mode)


        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        if self.down:
            x_out= F.conv1d(x, weight, stride=2)
        elif self.up:
            x_out =F.conv_transpose1d(x, weight, stride=2, padding=self.pad * 2 + 1)

        if self.mode_resample=="F":
            x_out=x_out.permute(0,2,1).contiguous()
            return x_out.view(shapeorig[0],-1,x_out.shape[-2], shapeorig[-1])
        else:
            return x_out.view(shapeorig[0],-1,shapeorig[2], x_out.shape[-1])

class MR_CQTDiff(nn.Module):
    """
        Main U-Net model based on the CQT
    """
    def __init__(self,
        representation="waveform",
        use_norm=True,
        depth=7,
        emb_dim=256,
        Ns= [64, 96 ,96, 128, 128,256, 256],
        checkpointing=[False,False,False,False,False,False,False],
        Stime= [2,2,2, 2, 2, 2, 2],
        Sfreq= [1,1,1, 1, 1, 1, 1],
        num_dils= [2,3,4,5,6,7,7],
        cqt=None, #it is a dictionary
        bottleneck_type= "res_dil_convs",
        num_bottleneck_layers= 1,
        sample_rate=22050,
        audio_len=184184,
        multiple_audio_lengths=None,
        in_features_context=1,
        F_dim=None,
        F_bins_per_oct=None,
        device=None):
        """
        Args:
            args (dictionary): hydra dictionary
            device: torch device ("cuda" or "cpu")
        """
        super(MR_CQTDiff, self).__init__()

        self.sample_rate=sample_rate

        self.device=device
        if device is None:
            print("No device specified, using cuda if available")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.depth=depth
        assert self.depth==sum(cqt.num_octs)

        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3)) #same as ADM, according to edm implementation
        init_zero = dict(init_mode='kaiming_uniform', init_weight=1e-7) #I think it is safer to initialize the last layer with a small weight, rather than zero. Breaking symmetry and all that. 

        self.emb_dim=emb_dim
        self.in_features_context=in_features_context
        if in_features_context==1:
            self.embedding = RFF_MLP_Block(emb_dim=emb_dim, init=init)
        else:
            self.embeddings = nn.ModuleList([RFF_MLP_Block(emb_dim=emb_dim, init=init) for _ in range(in_features_context)])
            self.emb_dim=emb_dim*in_features_context

        self.use_norm=use_norm

        self.bins_per_oct=cqt.bins_per_oct
        self.num_octs=cqt.num_octs

        self.bottleneck_type=bottleneck_type
        self.num_bottleneck_layers=num_bottleneck_layers

        self.representation=representation
        if self.representation=="waveform":
            if cqt.window=="kaiser":
                win=("kaiser",cqt.beta)
            else:
                win=cqt.window
    
            self.win=win
            self.sample_rate=sample_rate
    
            self.multiple_audio_lengths=multiple_audio_lengths
            assert self.multiple_audio_lengths is None
            
    
            fmax=[]
            for i in range(len(cqt.num_octs)):
                fmax.append(sample_rate * 2**(-1 - sum(cqt.num_octs[1+i:])))
    
            self.CQTransform = multi_CQT(numocts=cqt.num_octs,
                                binsoct=cqt.bins_per_oct,
                                fmax=fmax,
                                mode="oct",
                                window=self.win,
                                fs=sample_rate,
                                audio_len=audio_len,
                                dtype=torch.float32,
                                device=self.device)

        #Construct F_dim list using cqt.num_octs and cqt.bins_per_oct

        Nin=2

        #Encoder
        self.Ns= Ns
        self.Stime= Stime
        self.Sfreq= Sfreq

        assert F_dim is not None, "F_dim must be specified (it is the Frequency dimension of the CQTransform for each layer)"
        self.F_dim=F_dim
        self.F_bins_per_oct=F_bins_per_oct

        self.num_dils= num_dils #intuition: less dilations for the first layers and more for the deeper layers
        
        self.downsamplerT=UpDownResample(down=True, mode_resample="T")
        self.downsamplerF=UpDownResample(down=True, mode_resample="F")
        self.upsamplerT=UpDownResample(up=True, mode_resample="T")
        self.upsamplerF=UpDownResample(up=True, mode_resample="F")

        self.downs=nn.ModuleList([])
        self.middle=nn.ModuleList([])
        self.ups=nn.ModuleList([])

        self.checkpointing=checkpointing
        print(len(self.checkpointing), len(self.Ns))
        assert len(self.checkpointing)==len(self.Ns)

        
        for i in range(self.depth):
            if i==0:
                dim_in=self.Ns[i]
                dim_out=self.Ns[i]
            else:
                dim_in=self.Ns[i-1]
                dim_out=self.Ns[i]

            self.downs.append(
                               nn.ModuleList([
                                        ResnetBlock(Nin, dim_in, self.use_norm,num_dils=1, bias=False, kernel_size=(1,1), emb_dim=self.emb_dim, init=init, init_zero=init_zero),
                                        Conv2d(2, dim_out, kernel=(5,3), bias=False, **init),
                                        ResnetBlock(dim_in, dim_out, self.use_norm,num_dils=self.num_dils[i], bias=False , emb_dim=self.emb_dim, init=init, init_zero=init_zero, Fdim=self.F_dim[i])
                                        ]))

        if self.bottleneck_type=="res_dil_convs":
            for i in range(num_bottleneck_layers):
    
                self.middle.append(nn.ModuleList([
                                ResnetBlock(self.Ns[-1], 2, use_norm=self.use_norm,num_dils= 1,bias=False, kernel_size=(1,1), proj_place="after", emb_dim=self.emb_dim, init=init, init_zero=init_zero),
                                ResnetBlock(self.Ns[-1], self.Ns[-1], self.use_norm, num_dils=self.num_dils[-1], bias=False, emb_dim=self.emb_dim, init=init, init_zero=init_zero,
                                Fdim=self.F_dim[-1])
                                ]))
        else:
            raise NotImplementedError("bottleneck type not implemented")
                        
        for i in range(self.depth-1,-1,-1):

            if i==0:
                dim_in=self.Ns[i]*2
                dim_out=self.Ns[i]
            else:
                dim_in=self.Ns[i]*2
                dim_out=self.Ns[i-1]


            self.ups.append(nn.ModuleList(
                                        [
                                        ResnetBlock(dim_out, 2, use_norm=self.use_norm,num_dils= 1,bias=False, kernel_size=(1,1), proj_place="after", emb_dim=self.emb_dim, init=init, init_zero=init_zero),
                                        ResnetBlock(dim_in, dim_out, use_norm=self.use_norm,num_dils= self.num_dils[i], bias=False, emb_dim=self.emb_dim, init=init, init_zero=init_zero, Fdim=self.F_dim[i]),
                                        ]))

    def upsample(self, X, S_time, S_freq):
        
                if S_time==2:
                    X=self.upsamplerT(X)
                elif S_time==1:
                    pass
                else:
                    raise NotImplementedError("Only 1x and 2x downsampling is implemented")

                if S_freq==2:
                    X=self.upsamplerF(X)
                elif S_freq==1:
                    pass
                else:
                    raise NotImplementedError("Only 1x and 2x downsampling is implemented")

                return X

    def downsample(self, X, S_time, S_freq):
        
                if S_time==2:
                    X=self.downsamplerT(X)
                elif S_time==1:
                    pass
                else:
                    raise NotImplementedError("Only 1x and 2x downsampling is implemented")

                if S_freq==2:
                    X=self.downsamplerF(X)
                elif S_freq==1:
                    pass
                else:
                    raise NotImplementedError("Only 1x and 2x downsampling is implemented")

                return X


    def forward(self, inputs, sigma):
        """
        Args: 
            inputs (Tensor):  Input signal in time-domsin, shape (B,T)
            sigma (Tensor): noise levels,  shape (B,1)
        Returns:
            pred (Tensor): predicted signal in time-domain, shape (B,T)
        """
        #apply RFF embedding+MLP of the noise level
        sigma=sigma.to(self.device)

        if self.in_features_context==1:
            sigma = self.embedding(sigma)
        else:
            z = list()
            for i, embedding in enumerate(self.embeddings):
                z.append(embedding(sigma[:, i:i+1]))
            sigma = torch.cat(z, dim=-1)
            del z

        if self.representation=="waveform":
            original_shape=inputs.shape
            X_list =self.CQTransform.fwd(inputs) 
        elif self.representation=="CQT":
            X_list=inputs

        X_list_out=X_list

        for i in range(len(X_list)):
            x=X_list[i]

        hs=[]
        for i,modules in enumerate(self.downs):
            if i <(self.depth-1):
                Stime_i=self.Stime[i]
                Sfreq_i=self.Sfreq[i]

            if i <=(self.depth-1):

                C=X_list[-1-i]#get the corresponding CQT octave

                C=C.squeeze(1)
                C=torch.view_as_real(C)
                C=C.permute(0,3,1,2).contiguous() # call contiguous() here?
                C2=C
                    
                init_block, pyr_down_proj, ResBlock=modules
                C2=init_block(C2,sigma)
            else:
                pyr_down_proj, ResBlock=modules
            
            if i==0:
                X=C2 #starting the main signal path
                pyr=self.downsamplerT(C) #starting the auxiliary path
            elif i<(self.depth-1):
                C=self.downsample(C, Stime_i, Sfreq_i) #downsample the CQT octave
                pyr=self.downsample(pyr, Stime_i, Sfreq_i) #downsample the auxiliary path
                pyr=torch.cat((C,pyr), dim=2) #concatenate the CQT octave with the auxiliary path

                X=torch.cat((C2,X),dim=2) #updating the main signal path with the new octave
            elif i==(self.depth-1):# last layer
                pyr=torch.cat((C,pyr), dim=2) #no downsampling in the last layer
                X=torch.cat((C2,X),dim=2) #updating the main signal path with the new octave
            else: #last layer
                pass

            X=ResBlock(X, sigma, self.checkpointing[i])
            hs.append(X)

            if i<(self.depth-1): 
                X=self.downsample(X, Stime_i, Sfreq_i)

            else: #last layer
                #no downsampling in the last layer
                pass

            #apply the residual connection
            X=(X+pyr_down_proj(pyr))/(2**0.5) #I'll my need to put that inside a combiner block??

        #middle layers
        if self.bottleneck_type=="res_dil_convs":
            for i in range(self.num_bottleneck_layers):
                OutBlock, ResBlock =self.middle[i]
                X=ResBlock(X, sigma)   
                Xout=OutBlock(X,sigma)


        for i,modules in enumerate(self.ups):
            j=len(self.ups) -i-1
            if j <=(self.depth-1):
                Sfreq_i=self.Sfreq[j-1]
                Stime_i=self.Stime[j-1]

            OutBlock,  ResBlock=modules

            skip=hs.pop()
            X=torch.cat((X,skip),dim=1)
            X=ResBlock(X, sigma, self.checkpointing[j])
            
            Xout=(Xout+OutBlock(X,sigma))/(2**0.5)


            if j<=(self.depth-1):
                X= X[:,:,self.F_bins_per_oct[i]::,:]
                Out, Xout= Xout[:,:,0:self.F_bins_per_oct[i],:], Xout[:,:,self.F_bins_per_oct[i]::,:]

                Out=Out.permute(0,2,3,1).contiguous() #call contiguous() here?
                Out=torch.view_as_complex(Out)

                #save output
                X_list_out[i]=Out.unsqueeze(1)

            elif j>(self.depth-1):
                print("We should not be here")
                pass

            if j>0 and j<=(self.depth-1):
                
                X=self.upsample(X, Stime_i, Sfreq_i) #call contiguous() here?
                Xout=self.upsample(Xout, Stime_i, Sfreq_i) #call contiguous() here?

        if self.representation=="waveform":
            pred_time=self.CQTransform.bwd(X_list_out)
            pred_time=pred_time[...,0:original_shape[-1]]

            assert pred_time.shape==original_shape, "bad shapes"
            return pred_time
        elif self.representation=="CQT":
            return X_list_out


class CropAddBlock(nn.Module):

    def forward(self,down_layer, x,  **kwargs):
        x1_shape = down_layer.shape
        x2_shape = x.shape

        height_diff = (x1_shape[2] - x2_shape[2]) // 2
        width_diff = (x1_shape[3] - x2_shape[3]) // 2


        down_layer_cropped = down_layer[:,
                                        :,
                                        height_diff: (x2_shape[2] + height_diff),
                                        width_diff: (x2_shape[3] + width_diff),:]
        x = torch.add(down_layer_cropped, x)
        return x

class CropConcatBlock(nn.Module):

    def forward(self, down_layer, x, **kwargs):
        x1_shape = down_layer.shape
        x2_shape = x.shape

        height_diff = (x1_shape[2] - x2_shape[2]) // 2
        width_diff = (x1_shape[3] - x2_shape[3]) // 2
        down_layer_cropped = down_layer[:,
                                        :,
                                        height_diff: (x2_shape[2] + height_diff),
                                        width_diff: (x2_shape[3] + width_diff)]
        x = torch.cat((down_layer_cropped, x),1)
        return x

