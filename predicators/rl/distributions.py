"""Custom distributions in addition to torch's existing ones.

This file is adapted from the MAPLE codebase (https://github.com/UT-
Austin-RPL/maple) by Nasiriany et. al.
"""
from collections import OrderedDict

import numpy as np
import torch
from torch.distributions import Bernoulli as TorchBernoulli
from torch.distributions import Beta as TorchBeta
from torch.distributions import Categorical
from torch.distributions import Distribution as TorchDistribution
from torch.distributions import Independent as TorchIndependent
from torch.distributions import Normal as TorchNormal
from torch.distributions import OneHotCategorical, kl_divergence
from torch.distributions.utils import _sum_rightmost

import predicators.rl.rl_utils as rtu


class Distribution(TorchDistribution):
    def sample_and_logprob(self):
        s = self.sample()
        log_p = self.log_prob(s)
        return s, log_p

    def rsample_and_logprob(self):
        s = self.rsample()
        log_p = self.log_prob(s)
        return s, log_p

    def mle_estimate(self):
        return self.mean

    def get_diagnostics(self):
        return {}


class TorchDistributionWrapper(Distribution):
    def __init__(self, distribution: TorchDistribution):
        self.distribution = distribution

    @property
    def batch_shape(self):
        return self.distribution.batch_shape

    @property
    def event_shape(self):
        return self.distribution.event_shape

    @property
    def arg_constraints(self):
        return self.distribution.arg_constraints

    @property
    def support(self):
        return self.distribution.support

    @property
    def mean(self):
        return self.distribution.mean

    @property
    def variance(self):
        return self.distribution.variance

    @property
    def stddev(self):
        return self.distribution.stddev

    def sample(self, sample_size=torch.Size()):
        return self.distribution.sample(sample_shape=sample_size)

    def rsample(self, sample_size=torch.Size()):
        return self.distribution.rsample(sample_shape=sample_size)

    def log_prob(self, value):
        return self.distribution.log_prob(value)

    def cdf(self, value):
        return self.distribution.cdf(value)

    def icdf(self, value):
        return self.distribution.icdf(value)

    def enumerate_support(self, expand=True):
        return self.distribution.enumerate_support(expand=expand)

    def entropy(self):
        return self.distribution.entropy()

    def perplexity(self):
        return self.distribution.perplexity()

    def __repr__(self):
        return 'Wrapped ' + self.distribution.__repr__()


class Delta(Distribution):
    """A deterministic distribution."""
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value.detach()

    def rsample(self):
        return self.value

    @property
    def mean(self):
        return self.value

    @property
    def variance(self):
        return 0

    @property
    def entropy(self):
        return 0


class Bernoulli(Distribution, TorchBernoulli):
    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(create_stats_ordered_dict(
            'probability',
            rtu.to_numpy(self.probs),
        ))
        return stats


class Independent(Distribution, TorchIndependent):
    def get_diagnostics(self):
        return self.base_dist.get_diagnostics()


class Beta(Distribution, TorchBeta):
    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(create_stats_ordered_dict(
            'alpha',
            rtu.to_numpy(self.concentration0),
        ))
        stats.update(create_stats_ordered_dict(
            'beta',
            rtu.to_numpy(self.concentration1),
        ))
        stats.update(create_stats_ordered_dict(
            'entropy',
            rtu.to_numpy(self.entropy()),
        ))
        return stats


class MultivariateDiagonalNormal(TorchDistributionWrapper):
    from torch.distributions import constraints
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}

    def __init__(self, loc, scale_diag, reinterpreted_batch_ndims=1):
        dist = Independent(TorchNormal(loc, scale_diag),
                           reinterpreted_batch_ndims=reinterpreted_batch_ndims)
        super().__init__(dist)

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(create_stats_ordered_dict(
            'mean',
            rtu.to_numpy(self.mean),
            # exclude_max_min=True,
        ))
        stats.update(create_stats_ordered_dict(
            'std',
            rtu.to_numpy(self.distribution.stddev),
        ))
        return stats

    def __repr__(self):
        return self.distribution.base_dist.__repr__()


@torch.distributions.kl.register_kl(TorchDistributionWrapper,
                                    TorchDistributionWrapper)
def _kl_mv_diag_normal_mv_diag_normal(p, q):
    return kl_divergence(p.distribution, q.distribution)

# Independent RV KL handling - https://github.com/pytorch/pytorch/issues/13545

@torch.distributions.kl.register_kl(TorchIndependent, TorchIndependent)
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)

class GaussianMixture(Distribution):
    def __init__(self, normal_means, normal_stds, weights):
        self.num_gaussians = weights.shape[1]
        self.normal_means = normal_means
        self.normal_stds = normal_stds
        self.normal = MultivariateDiagonalNormal(normal_means, normal_stds)
        self.normals = [MultivariateDiagonalNormal(normal_means[:, :, i], normal_stds[:, :, i]) for i in range(self.num_gaussians)]
        self.weights = weights
        self.categorical = OneHotCategorical(self.weights[:, :, 0])

    def log_prob(self, value, ):
        log_p = [self.normals[i].log_prob(value) for i in range(self.num_gaussians)]
        log_p = torch.stack(log_p, -1)
        log_p = log_p.sum(dim=1)
        log_weights = torch.log(self.weights[:, :, 0])
        lp = log_weights + log_p
        m = lp.max(dim=1)[0]  # log-sum-exp numerical stability trick
        log_p_mixture = m + torch.log(torch.exp(lp - m).sum(dim=1))
        return log_p_mixture

    def sample(self):
        z = self.normal.sample().detach()
        c = self.categorical.sample()[:, :, None]
        s = torch.matmul(z, c)
        return torch.squeeze(s, 2)

    def rsample(self):
        z = (
                self.normal_means +
                self.normal_stds *
                MultivariateDiagonalNormal(
                    rtu.zeros(self.normal_means.size()),
                    rtu.ones(self.normal_stds.size())
                ).sample()
        )
        z.requires_grad_()
        c = self.categorical.sample()[:, :, None]
        s = torch.matmul(z, c)
        return torch.squeeze(s, 2)

    def mle_estimate(self):
        """Return the mean of the most likely component.

        This often computes the mode of the distribution, but not
        always.
        """
        c = rtu.zeros(self.weights.shape[:2])
        ind = torch.argmax(self.weights, dim=1) # [:, 0]
        c.scatter_(1, ind, 1)
        s = torch.matmul(self.normal_means, c[:, :, None])
        return torch.squeeze(s, 2)

    def __repr__(self):
        s = "GaussianMixture(normal_means=%s, normal_stds=%s, weights=%s)"
        return s % (self.normal_means, self.normal_stds, self.weights)


epsilon = 0.001


class GaussianMixtureFull(Distribution):
    def __init__(self, normal_means, normal_stds, weights):
        self.num_gaussians = weights.shape[-1]
        self.normal_means = normal_means
        self.normal_stds = normal_stds
        self.normal = MultivariateDiagonalNormal(normal_means, normal_stds)
        self.normals = [MultivariateDiagonalNormal(normal_means[:, :, i], normal_stds[:, :, i]) for i in range(self.num_gaussians)]
        self.weights = (weights + epsilon) / (1 + epsilon * self.num_gaussians)
        assert (self.weights > 0).all()
        self.categorical = Categorical(self.weights)

    def log_prob(self, value, ):
        log_p = [self.normals[i].log_prob(value) for i in range(self.num_gaussians)]
        log_p = torch.stack(log_p, -1)
        log_weights = torch.log(self.weights)
        lp = log_weights + log_p
        m = lp.max(dim=2, keepdim=True)[0]  # log-sum-exp numerical stability trick
        log_p_mixture = m + torch.log(torch.exp(lp - m).sum(dim=2, keepdim=True))
        raise NotImplementedError("from Vitchyr: idk what the point is of "
                                  "this class, so I didn't bother updating "
                                  "this, but log_prob should return something "
                                  "of shape [batch_size] and not [batch_size, "
                                  "1] to be in accordance with the "
                                  "torch.distributions.Distribution "
                                  "interface.")
        return torch.squeeze(log_p_mixture, 2)

    def sample(self):
        z = self.normal.sample().detach()
        c = self.categorical.sample()[:, :, None]
        s = torch.gather(z, dim=2, index=c)
        return s[:, :, 0]

    def rsample(self):
        z = (
                self.normal_means +
                self.normal_stds *
                MultivariateDiagonalNormal(
                    rtu.zeros(self.normal_means.size()),
                    rtu.ones(self.normal_stds.size())
                ).sample()
        )
        z.requires_grad_()
        c = self.categorical.sample()[:, :, None]
        s = torch.gather(z, dim=2, index=c)
        return s[:, :, 0]

    def mle_estimate(self):
        """Return the mean of the most likely component.

        This often computes the mode of the distribution, but not
        always.
        """
        ind = torch.argmax(self.weights, dim=2)[:, :, None]
        means = torch.gather(self.normal_means, dim=2, index=ind)
        return torch.squeeze(means, 2)

    def __repr__(self):
        s = "GaussianMixture(normal_means=%s, normal_stds=%s, weights=%s)"
        return s % (self.normal_means, self.normal_stds, self.weights)


class TanhNormal(Distribution):
    """Represent distribution of X where X ~ tanh(Z) Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6, prefix=''):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = MultivariateDiagonalNormal(normal_mean, normal_std)
        self.epsilon = epsilon

        self.prefix = prefix
        if len(prefix) > 0:
            self.prefix = '_' + self.prefix

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def _log_prob_from_pre_tanh(self, pre_tanh_value):
        """Adapted from https://github.com/tensorflow/probability/blob/master/t
        ensorflow_probability/python/bijectors/tanh.py#L73.

        This formula is mathematically equivalent to log(1 - tanh(x)^2).

        Derivation:

        log(1 - tanh(x)^2)
         = log(sech(x)^2)
         = 2 * log(sech(x))
         = 2 * log(2e^-x / (e^-2x + 1))
         = 2 * (log(2) - x - log(e^-2x + 1))
         = 2 * (log(2) - x - softplus(-2x))

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        log_prob = self.normal.log_prob(pre_tanh_value)
        correction = - 2. * (
            rtu.from_numpy(np.log([2.]))
            - pre_tanh_value
            - torch.nn.functional.softplus(-2. * pre_tanh_value)
        ).sum(dim=-1)
        return log_prob + correction

    def log_prob(self, value, pre_tanh_value=None):
        if pre_tanh_value is None:
            # errors or instability at values near 1
            value = torch.clamp(value, -0.999999, 0.999999)
            pre_tanh_value = torch.log(1+value) / 2 - torch.log(1-value) / 2
        return self._log_prob_from_pre_tanh(pre_tanh_value)

    def rsample_with_pretanh(self):
        z = (
                self.normal_mean +
                self.normal_std *
                MultivariateDiagonalNormal(
                    rtu.zeros(self.normal_mean.size()),
                    rtu.ones(self.normal_std.size())
                ).sample()
        )
        return torch.tanh(z), z

    def sample(self):
        """Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for
        discussion.
        """
        value, pre_tanh_value = self.rsample_with_pretanh()
        return value.detach()

    def rsample(self):
        """Sampling in the reparameterization case."""
        value, pre_tanh_value = self.rsample_with_pretanh()
        return value

    def sample_and_logprob(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        value, pre_tanh_value = value.detach(), pre_tanh_value.detach()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    def rsample_and_logprob(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    @property
    def mean(self):
        return torch.tanh(self.normal_mean)

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(create_stats_ordered_dict(
            'normal{}/mean'.format(self.prefix),
            rtu.to_numpy(self.mean),
        ))
        stats.update(create_stats_ordered_dict(
            'normal{}/std'.format(self.prefix),
            rtu.to_numpy(self.normal_std),
            exclude_max_min=False,
        ))
        stats.update(create_stats_ordered_dict(
            'normal{}/log_std'.format(self.prefix),
            rtu.to_numpy(torch.log(self.normal_std)),
        ))
        return stats

    def get_actions_and_logprobs(self):
        value, log_p = self.rsample_and_logprob()
        bs, action_dim = value.shape[0], value.shape[-1]
        expansion_factor = 1
        return (
            expansion_factor,
            value.reshape((bs, -1, action_dim)),
            log_p.reshape((bs, -1))
        )

class Softmax(Distribution):
    def __init__(self, logits, one_hot=False, prefix=''):
        self.logits = logits
        self.check_logits()

        self.one_hot = one_hot
        if one_hot:
            self.distr = torch.distributions.one_hot_categorical.OneHotCategorical(
                logits=self.logits,
            )
        else:
            self.distr = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
                temperature=rtu.ones(1),
                logits=self.logits,
            )

        self.prefix = prefix
        if len(prefix) > 0:
            self.prefix = '_' + self.prefix

    def check_logits(self):
        ### check logits are valid
        nans = torch.isnan(self.logits)
        infs = torch.isinf(self.logits)
        num_nans = rtu.to_numpy(torch.sum(nans))
        num_infs = rtu.to_numpy(torch.sum(infs))
        if num_nans > 0:
            print("WARNING! num nans:", num_nans)
        if num_infs > 0:
            print("WARNING! num infs:", num_infs)

    def sample_n(self, n):
        raise NotImplementedError

    def log_prob(self, value):
        return self.distr.log_prob(value)

    def print_logit_info(self):
        torch.set_printoptions(profile="full")
        print("logits:", self.logits[:100])
        torch.set_printoptions(profile="default")

    def sample(self):
        """Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for
        discussion.
        """
        try:
            if not self.one_hot:
                value = self.distr.rsample()
                return value.detach()
            else:
                return self.distr.sample()
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print("exception!", str(e))
            self.print_logit_info()
            exit()

    def rsample(self):
        """Sampling in the reparameterization case."""
        try:
            if not self.one_hot:
                return self.distr.rsample()
            else:
                return self.distr.sample()
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print("exception!", str(e))
            self.print_logit_info()
            exit()

    def sample_and_logprob(self):
        value = self.sample()
        log_p = self.log_prob(value)
        return value, log_p

    def rsample_and_logprob(self):
        value = self.rsample()
        log_p = self.log_prob(value)
        return value, log_p

    @property
    def mean(self):
        assert self.logits.dim() == 2
        max_idx = torch.argmax(self.logits, dim=1)
        one_hot = torch.nn.functional.one_hot(max_idx, num_classes=self.logits.shape[1])
        return one_hot

    def get_diagnostics(self):
        stats = OrderedDict()
        logits_np = rtu.to_numpy(self.logits)
        stats.update(create_stats_ordered_dict(
            'softmax{}/logit'.format(self.prefix),
            logits_np,
            exclude_max_min=False,
        ))
        stats.update(create_stats_ordered_dict(
            'softmax{}/logit_std'.format(self.prefix),
            np.std(logits_np, axis=1),
        ))
        stats.update(create_stats_ordered_dict(
            'softmax{}/logit_range'.format(self.prefix),
            np.max(logits_np, axis=1) - np.min(logits_np, axis=1),
        ))

        logits_sorted_np = np.sort(logits_np, axis=1)
        logits_sorted_np -= np.min(logits_np, axis=1).reshape((-1, 1))
        for i in range(logits_sorted_np.shape[1]):
            stats.update(create_stats_ordered_dict(
                'softmax{}/logit_{}'.format(self.prefix, i),
                logits_sorted_np[:,i],
            ))

        return stats

    def get_actions_and_logprobs(self):
        if self.one_hot:
            logits_shape = self.logits.shape
            bs, action_dim = logits_shape[0], logits_shape[-1]
            tile_dim = torch.numel(self.logits) // bs // action_dim

            value = torch.eye(action_dim).to(rtu.device)
            value = value.unsqueeze(0).repeat(bs, tile_dim, 1)
            log_p = torch.log(self.distr.probs)

            expansion_factor = action_dim
        else:
            value, log_p = self.rsample_and_logprob()
            bs, action_dim = value.shape[0], value.shape[-1]
            expansion_factor = 1

        return (
            expansion_factor,
            value.reshape((bs, -1, action_dim)),
            log_p.reshape((bs, -1))
        )

class HybridDistribution(Distribution):
    def __init__(self, rev_order):
        super().__init__()

        self.rev_order = rev_order

    def concat_values(self, value1, value2):
        if self.rev_order:
            return torch.cat([value2, value1], dim=-1)
        else:
            return torch.cat([value1, value2], dim=-1)

    def expand_tensors(self, tensor_list, expand_factor, interleave):
        for i in range(len(tensor_list)):
            tensor = tensor_list[i]
            if interleave:
                tensor = torch.repeat_interleave(
                    tensor,
                    expand_factor,
                    dim=1
                )
            else:
                repeat_dims = [1] * tensor.dim()
                repeat_dims[1] = expand_factor
                tensor = tensor.repeat(*repeat_dims)
            tensor_list[i] = tensor
        return tensor_list

class ConcatDistribution(HybridDistribution):
    def __init__(self, distr1, distr2, rev_order=False):
        super().__init__(rev_order=rev_order)

        self.distr1 = distr1
        self.distr2 = distr2

    def sample_n(self, n):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError

    def sample(self):
        """Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for
        discussion.
        """
        value1 = self.distr1.sample()
        value2 = self.distr2.sample()
        return self.concat_values(value1, value2)

    def rsample(self):
        """Sampling in the reparameterization case."""
        value1 = self.distr1.rsample()
        value2 = self.distr2.rsample()
        return self.concat_values(value1, value2)

    def sample_and_logprob(self):
        value1, *log_p1 = self.distr1.sample_and_logprob()
        value2, *log_p2 = self.distr2.sample_and_logprob()
        value = self.concat_values(value1, value2)
        if self.rev_order:
            return value, *log_p2, *log_p1
        else:
            return value, *log_p1, *log_p2

    def rsample_and_logprob(self):
        value1, *log_p1 = self.distr1.rsample_and_logprob()
        value2, *log_p2 = self.distr2.rsample_and_logprob()
        value = self.concat_values(value1, value2)
        if self.rev_order:
            return value, *log_p2, *log_p1
        else:
            return value, *log_p1, *log_p2

    @property
    def mean(self):
        return self.concat_values(self.distr1.mean, self.distr2.mean)

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(self.distr1.get_diagnostics())
        stats.update(self.distr2.get_diagnostics())
        return stats

    def get_actions_and_logprobs(self):
        ef1, value1, *log_p1 = self.distr1.get_actions_and_logprobs()
        ef2, value2, *log_p2 = self.distr2.get_actions_and_logprobs()

        value1, *log_p1 =  self.expand_tensors(
            tensor_list=[value1, *log_p1],
            expand_factor=ef2,
            interleave=(not self.rev_order),
        )
        value2, *log_p2 =  self.expand_tensors(
            tensor_list=[value2, *log_p2],
            expand_factor=ef1,
            interleave=self.rev_order,
        )

        value = self.concat_values(value1, value2)
        ef = ef1 * ef2
        if self.rev_order:
            return ef, value, *log_p2, *log_p1
        else:
            return ef, value, *log_p1, *log_p2

class DistributionList(Distribution):
    def __init__(self, distr1, distr2, rev_order=False):
        super().__init__(rev_order=rev_order)

        self.distr1 = distr1
        self.distr2 = distr2

class HierarchicalDistribution(HybridDistribution):
    def __init__(self, distr1, distr2_cond_fn, rev_order=False):
        super().__init__(rev_order=rev_order)

        self.distr1 = distr1
        self.distr2_cond_fn = distr2_cond_fn
        self.distr2_tmp = None

    def sample_n(self, n):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError

    def sample(self):
        """Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for
        discussion.
        """
        value1 = self.distr1.sample()
        distr2 = self.distr2_cond_fn(value1)
        self.distr2_tmp = distr2
        value2 = distr2.sample()
        return self.concat_values(value1, value2)

    def rsample(self):
        """Sampling in the reparameterization case."""
        value1 = self.distr1.rsample()
        distr2 = self.distr2_cond_fn(value1)
        self.distr2_tmp = distr2
        value2 = distr2.rsample()
        return self.concat_values(value1, value2)

    def sample_and_logprob(self):
        value1, *log_p1 = self.distr1.sample_and_logprob()
        distr2 = self.distr2_cond_fn(value1)
        self.distr2_tmp = distr2
        value2, *log_p2 = distr2.sample_and_logprob()
        value = self.concat_values(value1, value2)
        if self.rev_order:
            return value, *log_p2, *log_p1
        else:
            return value, *log_p1, *log_p2

    def rsample_and_logprob(self):
        value1, *log_p1 = self.distr1.rsample_and_logprob()
        distr2 = self.distr2_cond_fn(value1)
        self.distr2_tmp = distr2
        value2, *log_p2 = distr2.rsample_and_logprob()
        value = self.concat_values(value1, value2)
        if self.rev_order:
            return value, *log_p2, *log_p1
        else:
            return value, *log_p1, *log_p2

    @property
    def mean(self):
        value1 = self.distr1.mean
        distr2 = self.distr2_cond_fn(value1)
        self.distr2_tmp = distr2
        value2 = distr2.mean
        return self.concat_values(value1, value2)

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(self.distr1.get_diagnostics())
        if self.distr2_tmp is not None:
            stats.update(self.distr2_tmp.get_diagnostics())
        return stats

    def get_actions_and_logprobs(self):
        ef1, value1, *log_p1 = self.distr1.get_actions_and_logprobs()
        distr2 = self.distr2_cond_fn(value1)
        ef2, value2, *log_p2 = distr2.get_actions_and_logprobs()

        value1, *log_p1 =  self.expand_tensors(
            tensor_list=[value1, *log_p1],
            expand_factor=ef2,
            interleave=True,
        )
        value2, *log_p2 =  self.expand_tensors(
            tensor_list=[value2, *log_p2],
            expand_factor=1,
            interleave=False,
        )
        value = self.concat_values(value1, value2)
        ef = ef1 * ef2
        if self.rev_order:
            return ef, value, *log_p2, *log_p1
        else:
            return ef, value, *log_p1, *log_p2


class DistributionConcatValue(HybridDistribution):
    def __init__(self, value_fixed, distr, rev_order=False):
        super().__init__(rev_order=rev_order)

        self.value_fixed = value_fixed
        self.distr = distr

    def sample_n(self, n):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError

    def sample(self):
        """Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for
        discussion.
        """
        value_distr = self.distr.sample()
        return self.concat_values(self.value_fixed, value_distr)

    def rsample(self):
        """Sampling in the reparameterization case."""
        value_distr = self.distr.rsample()
        return self.concat_values(self.value_fixed, value_distr)

    def sample_and_logprob(self):
        value_distr, *log_p = self.distr.sample_and_logprob()
        value = self.concat_values(self.value_fixed, value_distr)
        return value, *log_p

    def rsample_and_logprob(self):
        value_distr, *log_p = self.distr.rsample_and_logprob()
        value = self.concat_values(self.value_fixed, value_distr)
        return value, *log_p

    @property
    def mean(self):
        value_distr = self.distr.mean
        return self.concat_values(self.value_fixed, value_distr)

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(self.distr.get_diagnostics())
        return stats

    def get_actions_and_logprobs(self):
        ef, value_distr, *log_p = self.distr.get_actions_and_logprobs()

        bs, action_dim = self.value_fixed.shape[0], self.value_fixed.shape[-1]
        value_fixed = self.value_fixed.reshape((bs, -1, action_dim))
        value_fixed = self.expand_tensors(
            tensor_list=[value_fixed],
            expand_factor=ef,
            interleave=(not self.rev_order),
        )[0]

        value = self.concat_values(value_fixed, value_distr)
        return ef, value, *log_p