import torch


class EDM():
    """
        Definition of the diffusion parameterization, following ( Karras et al., "Elucidating...", 2022). 
        This includes only the utilities needed for training, not for sampling.
    """

    def __init__(self,
        type,
        sde_hp):

        self.type = type
        self.sde_hp = sde_hp

        self.sigma_data = self.sde_hp.sigma_data #depends on the training data!! precalculated variance of the dataset
        self.sigma_min = self.sde_hp.sigma_min
        self.sigma_max = self.sde_hp.sigma_max
        self.rho = self.sde_hp.rho

        try:
            self.max_t= self.sde_hp.max_sigma
        except Exception as e:
            print(e)
            print("max_sigma not defined, please add it. It should be the highest sigma value seen during training")

    def sample_time_training(self, N):
        """
        For training, getting t according to a similar criteria as sampling. Simpler and safer to what Karras et al. did
        Args:
            N (int): batch size
        """
        a = torch.rand(N)
        t = (self.sigma_max**(1/self.rho) +a *(self.sigma_min**(1/self.rho) - self.sigma_max**(1/self.rho)))**self.rho

        return t

    def sample_prior(self, shape=None, t=None):
        """
        Just sample some gaussian noise, nothing more
        Args:
            shape (tuple): shape of the noise to sample, something like (B,T)
        """
        assert shape is not None
        if t is not None:
            n = torch.randn(shape).to(t.device) * t
        else:
            n = torch.randn(shape)
        return n

    def cskip(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        
        """
        return self.sigma_data**2 *(sigma**2+self.sigma_data**2)**-1

    def cout(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return sigma*self.sigma_data* (self.sigma_data**2+sigma**2)**(-0.5)

    def cin(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (self.sigma_data**2+sigma**2)**(-0.5)

    def cnoise(self, sigma):
        """
        preconditioning of the noise embedding
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (1/4)*torch.log(sigma)

    def lambda_w(self, sigma):
        """
        Score matching loss weighting
        """
        return (sigma*self.sigma_data)**(-2) * (self.sigma_data**2+sigma**2)
        
    def Tweedie2score(self, tweedie, xt, t, *args, **kwargs):
        return (tweedie - self._mean(xt, t)) / self._std(t)**2

    def score2Tweedie(self, score, xt, t, *args, **kwargs):
        return self._std(t)**2 * score + self._mean(xt, t)

    def _mean(self, x, t):
        return x
    
    def _std(self, t):
        return t
    
    def _ode_integrand(self, x, t, score):
        return -t * score
    

    def denoiser(self, xn , net, t, *args, **kwargs):
        """
        This method does the whole denoising step, which implies applying the model and the preconditioning
        Args:
            x (Tensor): shape: (B,1,T) Intermediate noisy latent to denoise
            model (nn.Module): Model of the denoiser
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        sigma = self._std(t).unsqueeze(-1)
        sigma = sigma.view(*sigma.size(), *(1,)*(xn.ndim - sigma.ndim))

        cskip = self.cskip(sigma)
        cout = self.cout(sigma)
        cin = self.cin(sigma)
        cnoise = self.cnoise(sigma.squeeze())

        #check if cnoise is a scalar, if so, repeat it
        if len(cnoise.shape) == 0:
            cnoise = cnoise.repeat(xn.shape[0],).unsqueeze(-1)
        else:
            cnoise = cnoise.view(xn.shape[0],).unsqueeze(-1)

        x_hat=cskip * xn + cout * net((cin * xn).to(torch.float32), cnoise.to(torch.float32)).to(xn.dtype)  #this will crash because of broadcasting problems, debug later!

        try:
            if net.CQTransform is not None:
                x_hat=net.CQTransform.apply_hpf_DC(x_hat)
        except:
            pass

        return x_hat
        

    def prepare_train_preconditioning(self, x, t,n=None, *args, **kwargs):
        mu, sigma = self._mean(x, t), self._std(t).unsqueeze(-1)
        sigma = sigma.view(*sigma.size(), *(1,)*(x.ndim - sigma.ndim))
        if n is None:
            n=self.sample_prior(shape=x.shape).to(x.device)
        x_perturbed = mu + sigma *n

        cskip = self.cskip(sigma)
        cout = self.cout(sigma)
        cin = self.cin(sigma)
        cnoise = self.cnoise(sigma.squeeze())

        #check if cnoise is a scalar, if so, repeat it
        if len(cnoise.shape) == 0:
            cnoise = cnoise.repeat(x.shape[0],)
        else:
            cnoise = cnoise.view(x.shape[0],)

        target = 1/cout * (x - cskip * x_perturbed)

        return cin * x_perturbed, target, cnoise

    def loss_fn(self, net, x,n=None, *args, **kwargs):
        """
        Loss function, which is the mean squared error between the denoised latent and the clean latent
        Args:
            net (nn.Module): Model of the denoiser
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        t = self.sample_time_training(x.shape[0]).to(x.device)

        input, target, cnoise = self.prepare_train_preconditioning(x, t, n=n)

        if len(cnoise.shape) == 1:
            #dirty patch
            cnoise = cnoise.unsqueeze(-1)

        if input.ndim==2:
            input=input.unsqueeze(1)

        estimate = net(input, cnoise)
        
        if target.ndim==2 and estimate.ndim==3:
            estimate=estimate.squeeze(1)

        error = estimate - target

        return error**2, self._std(t)
