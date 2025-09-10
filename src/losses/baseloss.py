"""
https://github.com/facebookresearch/disentangling-correlated-factors/blob/main/dent/losses/baseloss.py
"""

import abc

class BaseLoss(abc.ABC):
    """
    Base class for losses.

    Parameters
    ----------
    rec_dist: {"bernoulli", "gaussian", "laplace"}, optional
        Reconstruction distribution of the likelihood on each pixel.
        Implicitly defines the reconstruction loss. Bernoulli corresponds to a
        binary cross entropy (bse), Gaussian corresponds to MSE, Laplace
        corresponds to L1.

    mode: str
        Determines how the loss operates: if 'post_forward', it only takes standard model forward outputs/data points
        and returns a loss. If 'pre_forward', it takes in the model and performs respective forward computations itself.
        If 'optimizes_internally', takes model, does forward computations AND backward updates.
    """
    def __init__(self, rec_dist, mode='post_forward', **kwargs):
        self.n_train_steps = 0
        self.rec_dist = rec_dist
        self.mode = mode

    @abc.abstractmethod
    def __call__(self, data, reconstructions, stats_qzx, is_train, **kwargs):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Shape : (batch_size, n_chan,
            height, width).

        reconstructions : torch.Tensor
            Reconstructed data. Shape : (batch_size, n_chan, height, width).

        stats_qzx : torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).

        is_train : bool
            Whether currently in train mode.

        storer : dict
            Dictionary in which to store important variables for vizualisation.

        kwargs:
            Loss specific arguments
        """

    def attrs_to_chkpt(self):
        """Return dict of attributes to retain in checkpoint.
        """
        return {'n_train_steps': self.n_train_steps}

    def _pre_call(self, is_train):
        if is_train:
            self.n_train_steps += 1
