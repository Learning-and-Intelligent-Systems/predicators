"""Various 'Trainers' that do all the training necessary for
RL approaches.

TODO: Deprecate this file and fold + simplify the training procedure into a more
standard file.

This file is adapted from the MAPLE codebase
(https://github.com/UT-Austin-RPL/maple) by Nasiriany et. al.
"""

from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim

from predicators.rl.policies import PAMDPPolicy

import predicators.rl.rl_utils as rtu
import gtimer as gt

SACLosses = namedtuple(
    'SACLosses',
    'policy_loss qf1_loss qf2_loss alpha_loss',
)

class SACTrainer:
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            target_entropy_config=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        if target_entropy_config is None:
            self.target_entropy_config = {}
        else:
            self.target_entropy_config = target_entropy_config

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            else:
                self.target_entropy = target_entropy

            self.target_entropy_config.update(dict(
                later=self.target_entropy,
            ))

            self.log_alpha = rtu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = torch.nn.MSELoss()
        self.vf_criterion = torch.nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train_from_torch(self, batch):
        gt.blank_stamp()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            losses.alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        self.qf2_optimizer.step()

        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp('sac training', unique=False)

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        rtu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        rtu.soft_update_from_to(
            self.qf2, self.target_qf2, self.soft_target_tau
        )

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[SACLosses, OrderedDict]:
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['QF1 Loss'] = np.mean(rtu.get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(rtu.get_numpy(qf2_loss))
            eval_statistics['Policy Loss'] = np.mean(rtu.get_numpy(
                policy_loss
            ))
            eval_statistics.update(rtu.rtu.create_stats_ordered_dict(
                'Q1 Predictions',
                rtu.get_numpy(q1_pred),
            ))
            eval_statistics.update(rtu.rtu.create_stats_ordered_dict(
                'Q2 Predictions',
                rtu.get_numpy(q2_pred),
            ))
            eval_statistics.update(rtu.rtu.create_stats_ordered_dict(
                'Q Targets',
                rtu.get_numpy(q_target),
            ))
            eval_statistics.update(rtu.rtu.create_stats_ordered_dict(
                'Log Pis',
                rtu.get_numpy(log_pi),
            ))
            policy_statistics = rtu.add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()

        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
        )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

        if not self.use_automatic_entropy_tuning:
            return

        config = self.target_entropy_config
        for k in config:
            assert k in ['init_epochs', 'init', 'later']
        init_epochs = config.get('init_epochs', 100)
        if epoch <= init_epochs:
            self.target_entropy_ent = config.get('init', config['later'])
        else:
            self.target_entropy_ent = config['later']

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    @property
    def optimizers(self):
        return [
            self.alpha_optimizer,
            self.qf1_optimizer,
            self.qf2_optimizer,
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )


SACHybridLosses = namedtuple(
    'SACLosses',
    'policy_loss qf1_loss qf2_loss alpha_s_loss alpha_p_loss',
)

class SACHybridTrainer(SACTrainer):
    def __init__(
            self,
            target_entropy_s=None,
            target_entropy_p=None,
            target_entropy_config=None,
            **sac_kwargs
    ):
        super().__init__(**sac_kwargs)

        optimizer_class = sac_kwargs.get('optimizer_class', optim.Adam)
        policy_lr = sac_kwargs.get('policy_lr', None)

        if target_entropy_config is None:
            self.target_entropy_config = {}
        else:
            self.target_entropy_config = target_entropy_config

        if self.use_automatic_entropy_tuning:
            one_hot_factor = self.target_entropy_config.get('one_hot_factor', 0.75)

            if target_entropy_s is not None:
                self.target_entropy_s = target_entropy_s
            else:
                if self.one_hot_s:
                    self.target_entropy_s = np.log(self.policy.action_dim_s) * one_hot_factor
                else:
                    self.target_entropy_s = -self.policy.action_dim_s

            if target_entropy_p is not None:
                self.target_entropy_p = target_entropy_p
            else:
                self.target_entropy_p = -self.policy.action_dim_p

            self.target_entropy_config.update(dict(
                later_s=self.target_entropy_s,
                later_p=self.target_entropy_p,
            ))

            self.log_alpha_s = rtu.zeros(1, requires_grad=True)
            self.log_alpha_p = rtu.zeros(1, requires_grad=True)

            self.alpha_s_optimizer = optimizer_class(
                [self.log_alpha_s],
                lr=policy_lr,
            )
            self.alpha_p_optimizer = optimizer_class(
                [self.log_alpha_p],
                lr=policy_lr,
            )

    def train_from_torch(self, batch):
        gt.blank_stamp()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """
        if self.use_automatic_entropy_tuning:
            if losses.alpha_s_loss is not None:
                self.alpha_s_optimizer.zero_grad()
                losses.alpha_s_loss.backward()
                self.alpha_s_optimizer.step()

            self.alpha_p_optimizer.zero_grad()
            losses.alpha_p_loss.backward()
            self.alpha_p_optimizer.step()

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        self.qf2_optimizer.step()

        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp('sac training', unique=False)

    @property
    def one_hot_s(self):
        if isinstance(self.policy, PAMDPPolicy):
            return self.policy.one_hot_s
        else:
            return False

    @property
    def one_hot(self):
        return self.one_hot_s is True

    def get_dist_dict(self, obs, one_hot=False):
        dist = self.policy(obs)
        dist_dict = {
            'bs': obs.shape[0],
            'dist': dist,
        }

        if one_hot:
            _, actions, *log_pi = dist.get_actions_and_logprobs()
        else:
            actions, *log_pi = dist.rsample_and_logprob()

        actions = actions.reshape((-1, actions.shape[-1]))
        for i in range(len(log_pi)):
            log_pi[i] = log_pi[i].reshape((-1, 1))

        assert len(log_pi) == 2
        dist_dict['log_pi_s'] = log_pi[0]
        dist_dict['log_pi_p'] = log_pi[1]

        if one_hot:
            obs = obs.repeat_interleave(
                actions.shape[0] // obs.shape[0],
                dim=0,
            )

        dist_dict['obs'] = obs
        dist_dict['actions'] = actions

        return dist_dict

    def reduce_tensor(self, loss, dd):
        if self.one_hot_s and 'log_pi_s' in dd:
            loss = loss * torch.exp(dd['log_pi_s'])
        loss = loss.reshape((dd['bs'], -1))
        return torch.sum(loss, dim=1).mean()

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[SACLosses, OrderedDict]:
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        dd = self.get_dist_dict(obs, one_hot=self.one_hot)
        log_pi_s = dd.get('log_pi_s', None)
        log_pi_p = dd['log_pi_p']

        if self.use_automatic_entropy_tuning:
            if log_pi_s is not None:
                alpha_s_loss = -self.log_alpha_s * self.reduce_tensor(
                    log_pi_s + self.target_entropy_s, dd
                ).detach()
                alpha_s = self.log_alpha_s.exp()
            else:
                alpha_s_loss = None
                alpha_s = None

            alpha_p_loss = -self.log_alpha_p * self.reduce_tensor(
                log_pi_p + self.target_entropy_p, dd
            ).detach()
            alpha_p = self.log_alpha_p.exp()
        else:
            if log_pi_s is not None:
                alpha_s_loss = 0
                alpha_s = 1
            else:
                alpha_s_loss = None
                alpha_s = None

            alpha_p_loss = 0
            alpha_p = 1

        q_new_actions = torch.min(
            self.qf1(dd['obs'], dd['actions']),
            self.qf2(dd['obs'], dd['actions']),
        )
        alpha_log_pi = alpha_p * log_pi_p
        if log_pi_s is not None:
            alpha_log_pi += (alpha_s * log_pi_s)

        policy_loss = alpha_log_pi - q_new_actions
        policy_loss = self.reduce_tensor(policy_loss, dd)

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        next_dist_dict = self.get_dist_dict(next_obs)
        new_next_actions = next_dist_dict['actions']
        new_log_pi_s = next_dist_dict.get('log_pi_s', None)
        new_log_pi_p = next_dist_dict['log_pi_p']
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        )
        target_q_values = target_q_values - (alpha_p * new_log_pi_p)
        if new_log_pi_s is not None:
            target_q_values -= (alpha_s * new_log_pi_s)

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            dd = self.get_dist_dict(obs)

            eval_statistics['QF1 Loss'] = np.mean(rtu.get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(rtu.get_numpy(qf2_loss))
            eval_statistics['Policy Loss'] = np.mean(rtu.get_numpy(
                policy_loss
            ))
            eval_statistics.update(rtu.create_stats_ordered_dict(
                'Q1 Predictions',
                rtu.get_numpy(q1_pred),
            ))
            eval_statistics.update(rtu.create_stats_ordered_dict(
                'Q2 Predictions',
                rtu.get_numpy(q2_pred),
            ))
            eval_statistics.update(rtu.create_stats_ordered_dict(
                'Q Targets',
                rtu.get_numpy(q_target),
            ))

            if 'log_pi_s' in dd:
                eval_statistics.update(rtu.create_stats_ordered_dict(
                    'Log Pis S',
                    rtu.get_numpy(dd['log_pi_s']),
                ))
            eval_statistics.update(rtu.create_stats_ordered_dict(
                'Log Pis P',
                rtu.get_numpy(dd['log_pi_p']),
            ))

            policy_statistics = rtu.add_prefix(dd['dist'].get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                if 'log_pi_s' in dd:
                    eval_statistics['Alpha S'] = alpha_s.item()
                    eval_statistics['Alpha S Loss'] = alpha_s_loss.item()

                eval_statistics['Alpha P'] = alpha_p.item()
                eval_statistics['Alpha P Loss'] = alpha_p_loss.item()

        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_s_loss=alpha_s_loss,
            alpha_p_loss=alpha_p_loss,
        )

        return loss, eval_statistics

    @property
    def optimizers(self):
        return [
            self.alpha_s_optimizer,
            self.alpha_p_optimizer,
            self.qf1_optimizer,
            self.qf2_optimizer,
            self.policy_optimizer,
        ]

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

        if not self.use_automatic_entropy_tuning:
            return

        config = self.target_entropy_config
        for k in config:
            assert k in [
                'init_epochs', 'init_s', 'init_p',
                'later_s', 'later_p',
                'one_hot_factor',
            ]
        init_epochs = config.get('init_epochs', 100)
        if epoch <= init_epochs:
            self.target_entropy_s = config.get('init_s', config['later_s'])
            self.target_entropy_p = config.get('init_p', config['later_p'])
        else:
            self.target_entropy_s = config['later_s']
            self.target_entropy_p = config['later_p']
