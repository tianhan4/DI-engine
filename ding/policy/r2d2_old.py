import copy
import math
from collections import namedtuple
from typing import List, Dict, Any, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ding.model import model_wrap
from ding.rl_utils import q_nstep_td_data, q_nstep_td_error, q_nstep_td_error_with_rescale, get_nstep_return_data, \
    get_train_sample
from ding.torch_utils import Adam, to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import timestep_collate, default_collate, default_decollate
from .base_policy import Policy


@POLICY_REGISTRY.register('r2d2')
class R2D2Policy(Policy):
    r"""
    Overview:
        Policy class of R2D2, from paper `Recurrent Experience Replay in Distributed Reinforcement Learning` .
        R2D2 proposes that several tricks should be used to improve upon DRQN,
        namely some recurrent experience replay tricks such as burn-in.

    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      dqn            | RL policy register name, refer to      | This arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | This arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4  ``priority``         bool     False          | Whether use priority(PER)              | Priority sample,
                                                                                                 | update priority
        5  | ``priority_IS``    bool     False          | Whether use Importance Sampling Weight
           | ``_weight``                                | to correct biased update. If True,
                                                        | priority must be True.
        6  | ``discount_``      float    0.997,         | Reward's future discount factor, aka.  | May be 1 when sparse
           | ``factor``                  [0.95, 0.999]  | gamma                                  | reward env
        7  ``nstep``            int      3,             | N-step reward discount sum for target
                                         [3, 5]         | q_value estimation
        8  ``burnin_step``      int      2              | The timestep of burnin operation,
                                                        | which is designed to RNN hidden state
                                                        | difference caused by off-policy
        9  | ``learn.update``   int      1              | How many updates(iterations) to train  | This args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        10 | ``learn.batch_``   int      64             | The number of samples of an iteration
           | ``size``
        11 | ``learn.learning`` float    0.001          | Gradient step length of an iteration.
           | ``_rate``
        12 | ``learn.value_``   bool     True           | Whether use value_rescale function for
           | ``rescale``                                | predicted value
        13 | ``learn.target_``  int      100            | Frequence of target network update.    | Hard(assign) update
           | ``update_freq``
        14 | ``learn.ignore_``  bool     False          | Whether ignore done for target value   | Enable it for some
           | ``done``                                   | calculation.                           | fake termination env
        15 ``collect.n_sample`` int      [8, 128]       | The number of training samples of a    | It varies from
                                                        | call of collector.                     | different envs
        16 | ``collect.unroll`` int      1              | unroll length of an iteration          | In RNN, unroll_len>1
           | ``_len``
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='r2d2',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=True,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=True,
        # ==============================================================
        # The following configs are algorithm-specific
        # ==============================================================
        # (float) Reward's future discount factor, aka. gamma.
        discount_factor=0.997,
        # (int) N-step reward for target q_value estimation
        nstep=5,
        # (int) the timestep of burnin operation, which is designed to RNN hidden state difference
        # caused by off-policy
        burnin_step=2,
        # (int) the trajectory length to unroll the RNN network minus
        # the timestep of burnin operation
        unroll_len=80,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            update_per_collect=1,
            batch_size=64,
            learning_rate=0.0001,
            # (str) Optimizer type for ding.torch_utils.Adam: 'adam' (L2 penalty) or 'adamw' (decoupled).
            optim_type='adamw',
            # (float) Weight decay for optimizer (meaning depends on optim_type).
            weight_decay=1e-4,
            # (str|None) Gradient clipping type handled inside optimizer. Recommend 'clip_norm' for RNN stability.
            grad_clip_type='clip_norm',  # None|'clip_norm'|'clip_value'|'clip_momentum'|'clip_momentum_norm'
            # (float|None) Clip threshold (for clip_norm this is max_norm; for clip_value this is abs(grad) clip).
            grad_clip_value=10.0,
            # (float) Norm type used when grad_clip_type='clip_norm'
            grad_clip_norm_type=2.0,
            # (bool) If True, ignore TD loss and train as supervised learning on provided `metrics`.
            # This repo injects `metrics` into samples in env wrapper; the model outputs `pred_metrics`
            # from DiscreteHead, and we regress metrics for the taken action.
            supervised_metrics_only=False,
            # (float) Weight for the supervised metrics loss.
            metrics_loss_weight=1.0,
            # (dict) Periodic per-metric temperature calibration for heteroscedastic metrics.
            metrics_calibration=dict(
                enable=False, # 是否启用温度校准
                interval=200, # 校准间隔 100~500
                holdout_ratio=0.2, # 保留比例 0.1~0.2
                ema_decay=0.0, # EMA 衰减率 0.9~0.99
                min_log_t=-2.0, # 最小对数温度最小缩放约 0.13x
                max_log_t=2.0, # 最大对数温度 最大缩放约 7.38x
            ),
            # Preference w-augmentation for smoother utility behavior.
            metrics_w_aug=dict(
                enable=True, # 是否启用 preference w-augmentation
                num_samples=4, # 采样次数
                noise_std=0.05, # 噪声标准差
                normalize_w=True, # 是否归一化
                clamp_min=0.0, # 最小值
                loss_weight=0.1, # 损失权重
            ),
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            # target_update_freq=100,
            target_update_theta=0.001,
            # (bool) whether use value_rescale function for predicted value
            value_rescale=True,
            ignore_done=False,
        ),
        collect=dict(
            # NOTE it is important that don't include key n_sample here, to make sure self._traj_len=INF
            each_iter_n_sample=32,
            # `env_num` is used in hidden state, should equal to that one in env config.
            # User should specify this value in user config.
            env_num=None,
            # (str) Exploration type in collect. 'eps_greedy' uses eps passed into _forward_collect.
            # 'ucb' uses MC-dropout to estimate mean/std of per-action utility (logit),
            # then selects argmax(mean + ucb_beta * std), optionally mixed with eps random.
            exploration_type='eps_greedy',  # eps_greedy|ucb|thompson
            ucb_beta=1.0,
            mc_dropout_samples=8,
            # (float) Risk-averse penalty coefficient for aleatoric uncertainty of utility u = w·m.
            # If > 0 and the model returns `pred_metrics_log_std`, action score is penalized by `risk_alpha * sigma_u`.risk_alpha 的推荐起始值为 1.0，合理的调节范围在 0.5 到 2.0 之间。
            risk_alpha=0.0,
        ),
        eval=dict(
            # `env_num` is used in hidden state, should equal to that one in env config.
            # User should specify this value in user config.
            env_num=None,
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=10000, ),
        ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Init the learner model of R2D2Policy

        Arguments:
            .. note::

                The _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): The learning rate fo the optimizer
            - gamma (:obj:`float`): The discount factor
            - nstep (:obj:`int`): The num of n step return
            - value_rescale (:obj:`bool`): Whether to use value rescaled loss in algorithm
            - burnin_step (:obj:`int`): The num of step of burnin
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._metrics_dim = self._get_metrics_dim()
        # Optimizer, weight decay, and gradient clipping are configured from cfg.learn to make runs reproducible.
        learn_cfg = self._cfg.learn
        optim_type = str(getattr(learn_cfg, 'optim_type', 'adam')).lower()
        weight_decay = float(getattr(learn_cfg, 'weight_decay', 0.0))
        grad_clip_type = getattr(learn_cfg, 'grad_clip_type', None)
        clip_value = getattr(learn_cfg, 'grad_clip_value', None)
        clip_norm_type = float(getattr(learn_cfg, 'grad_clip_norm_type', 2.0))
        # ding.torch_utils.Adam asserts: if grad_clip_type is not None, clip_value must be provided.
        if grad_clip_type in [None, 'None', 'none', '']:
            grad_clip_type = None
            clip_value = None
        self._optimizer = Adam(
            self._model.parameters(),
            lr=float(self._cfg.learn.learning_rate),
            weight_decay=weight_decay,
            optim_type=optim_type,
            grad_clip_type=grad_clip_type,
            clip_value=clip_value,
            clip_norm_type=clip_norm_type,
        )
        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep
        self._burnin_step = self._cfg.burnin_step
        self._value_rescale = self._cfg.learn.value_rescale
        self._init_metrics_calibration()

        self._target_model = copy.deepcopy(self._model)
        # here we should not adopt the 'assign' mode of target network here because the reset bug
        # self._target_model = model_wrap(
        #     self._target_model,
        #     wrapper_name='target',
        #     update_type='assign',
        #     update_kwargs={'freq': self._cfg.learn.target_update_freq}
        # )
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_update_theta}
        )

        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
        )
        self._learn_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
        )
        self._learn_model = model_wrap(self._learn_model, wrapper_name='argmax_sample')
        self._learn_model.reset()
        self._target_model.reset()

    def _extract_obs_tensor(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, dict):
            if 'agent_state' in obs:
                return obs['agent_state']
            if 'obs' in obs:
                return obs['obs']
            # fallback to the first tensor-like value
            for v in obs.values():
                return v
        return obs

    def _get_metrics_dim(self) -> int:
        model_cfg = getattr(self._cfg, 'model', None)
        metrics_dim = int(getattr(model_cfg, 'metrics_dim', 4)) if model_cfg is not None else 4
        if metrics_dim <= 0:
            raise ValueError(f"metrics_dim must be positive, got {metrics_dim}")
        return metrics_dim

    def _extract_preference_w(self, obs_tensor: Any) -> torch.Tensor:
        metrics_dim = getattr(self, '_metrics_dim', self._get_metrics_dim())
        if not torch.is_tensor(obs_tensor):
            raise RuntimeError(f"Expected obs tensor, got: {type(obs_tensor)}")
        if obs_tensor.shape[-1] < metrics_dim:
            raise RuntimeError(
                f"Expected obs last dim >= {metrics_dim} to include preference w, got shape={tuple(obs_tensor.shape)}"
            )
        return obs_tensor[..., -metrics_dim:]

    def _calibration_scalar_dict(self) -> Dict[str, float]:
        calib_log_t = None if self._metrics_log_t is None else self._metrics_log_t.detach().cpu().tolist()
        out = {'metrics_calib_log_t': calib_log_t}
        for i in range(self._metrics_dim):
            log_t = 0.0
            if isinstance(calib_log_t, list) and len(calib_log_t) == self._metrics_dim:
                log_t = float(calib_log_t[i])
            out[f'metrics_calib_log_t_{i}'] = log_t
            out[f'metrics_calib_t_{i}'] = float(math.exp(log_t))
        return out

    def _init_metrics_calibration(self) -> None:
        cfg = getattr(self._cfg.learn, 'metrics_calibration', None)
        self._metrics_calibration_enabled = bool(cfg and getattr(cfg, 'enable', False))
        self._metrics_calibration_interval = int(getattr(cfg, 'interval', 200)) if cfg else 0
        self._metrics_calibration_holdout_ratio = float(getattr(cfg, 'holdout_ratio', 0.2)) if cfg else 0.0
        self._metrics_calibration_ema = float(getattr(cfg, 'ema_decay', 0.0)) if cfg else 0.0
        self._metrics_calibration_min_log_t = float(getattr(cfg, 'min_log_t', -2.0)) if cfg else -2.0
        self._metrics_calibration_max_log_t = float(getattr(cfg, 'max_log_t', 2.0)) if cfg else 2.0
        self._metrics_calibration_step = 0
        if self._metrics_calibration_enabled:
            device = self._device if hasattr(self, '_device') else next(self._model.parameters()).device
            self._metrics_log_t = torch.zeros(self._metrics_dim, device=device)
        else:
            self._metrics_log_t = None

    def _apply_metrics_calibration(self, pred_log_std: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self._metrics_calibration_enabled or pred_log_std is None:
            return pred_log_std
        view_shape = [1] * (pred_log_std.dim() - 1) + [self._metrics_log_t.numel()]
        return pred_log_std + self._metrics_log_t.view(*view_shape)

    def _maybe_calibrate_metrics(
        self,
        pred: torch.Tensor,
        pred_log_std: Optional[torch.Tensor],
        target: torch.Tensor,
    ) -> None:
        if not self._metrics_calibration_enabled or pred_log_std is None:
            return
        self._metrics_calibration_step += 1
        if self._metrics_calibration_interval <= 0:
            return
        if self._metrics_calibration_step % self._metrics_calibration_interval != 0:
            return
        with torch.no_grad():
            pred = pred.detach().reshape(-1, pred.shape[-1])
            pred_log_std = pred_log_std.detach().reshape(-1, pred_log_std.shape[-1])
            target = target.detach().reshape(-1, target.shape[-1])
            n = pred.shape[0]
            if n < 2:
                return
            holdout_ratio = float(self._metrics_calibration_holdout_ratio)
            if 0.0 < holdout_ratio < 1.0: #在线 Batch 内校准,随机采样
                k = max(1, int(n * holdout_ratio))
                idx = torch.randperm(n, device=pred.device)[:k]
                pred = pred[idx]
                pred_log_std = pred_log_std[idx]
                target = target[idx]
            # 我们希望校准后的方差能够等于真实误差，即：$V_{actual} \approx t^2 \cdot V_{pred}$ （其中 $t$ 就是我们要找的温度系数）。推导比率：$t^2 = \frac{V_{actual}}{V_{pred}}$ （也就是代码中的 ratio）。两边同时取自然对数：$\log(t^2) = \log(ratio)$。化简得到我们要的对数系数：$2 \log(t) = \log(ratio) \implies \log(t) = 0.5 \cdot \log(ratio)$。这就是 new_log_t = 0.5 * torch.log(ratio) 的由来！如果模型过于自信（实际误差 diff2 远大于 预测方差 var），ratio 就会大于 1，new_log_t 就是正数，在后续应用时就会把模型的标准差强行放大。
            diff2 = (target - pred) ** 2 # 实际误差平方 (真实方差)
            var = torch.exp(2.0 * pred_log_std) # 模型预测的方差
            ratio = (diff2 / (var + 1e-12)).mean(dim=0).clamp(min=1e-6)
            new_log_t = 0.5 * torch.log(ratio)
            new_log_t = new_log_t.clamp(
                min=self._metrics_calibration_min_log_t, max=self._metrics_calibration_max_log_t
            )
            if self._metrics_calibration_ema > 0.0:
                ema = float(self._metrics_calibration_ema)
                new_log_t = ema * self._metrics_log_t + (1.0 - ema) * new_log_t
            self._metrics_log_t.copy_(new_log_t)

    def _compute_metrics_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_log_std: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        cfg = self._cfg.learn
        loss_type = str(getattr(cfg, 'metrics_loss_type', 'mse')).lower()
        per_dim_weight = getattr(cfg, 'metrics_per_dim_weight', None)
        weight = None
        if per_dim_weight is not None:
            if len(per_dim_weight) != pred.shape[-1]:
                raise ValueError(
                    f"metrics_per_dim_weight length must match metrics dim {pred.shape[-1]}, got {len(per_dim_weight)}"
                )
            w = torch.tensor(per_dim_weight, device=pred.device, dtype=pred.dtype)
            view_shape = [1] * (pred.dim() - 1) + [w.numel()]
            weight = w.view(*view_shape)

        diff = pred - target
        if loss_type in ['huber', 'smooth_l1']:
            beta = float(getattr(cfg, 'metrics_huber_beta', 1.0))
            abs_diff = diff.abs()
            loss_elem = torch.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)
        elif loss_type in ['gaussian_nll', 'nll', 'gaussian_nll_huber']:
            if pred_log_std is None:
                raise KeyError("metrics_loss_type=gaussian_nll requires pred_metrics_log_std from the model head.")
            var = torch.exp(2.0 * pred_log_std)
            nll = 0.5 * (diff ** 2) / var + pred_log_std
            if loss_type == 'gaussian_nll_huber':
                beta = float(getattr(cfg, 'metrics_huber_beta', 1.0))
                huber_weight = float(getattr(cfg, 'metrics_huber_weight', 0.1))
                abs_diff = diff.abs()
                huber = torch.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)
                loss_elem = nll + huber_weight * huber
            else:
                loss_elem = nll
        else:
            # default to MSE
            loss_elem = diff ** 2

        if weight is not None:
            loss_elem = loss_elem * weight

        loss_per_sample = loss_elem.mean(dim=-1)
        if reduction == 'none':
            return loss_per_sample
        return loss_per_sample.mean()

    def _reduce_supervised_loss(
        self,
        loss_per_sample: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.to(device=loss_per_sample.device, dtype=loss_per_sample.dtype)
        if sample_weight is None:
            if mask is None:
                return loss_per_sample.mean()
            return (loss_per_sample * mask).sum() / (mask.sum() + 1e-6)

        weight = sample_weight.to(device=loss_per_sample.device, dtype=loss_per_sample.dtype)
        if weight.dim() == 0:
            weight = weight.view(1)
        if weight.shape[-1] != loss_per_sample.shape[-1]:
            raise ValueError(
                f"Expected sample_weight last dim to match batch dim {loss_per_sample.shape[-1]}, got {tuple(weight.shape)}"
            )
        while weight.dim() < loss_per_sample.dim():
            weight = weight.unsqueeze(0)

        weighted_loss = loss_per_sample * weight
        effective_weight = weight
        if mask is not None:
            weighted_loss = weighted_loss * mask
            effective_weight = effective_weight * mask
        return weighted_loss.sum() / (effective_weight.sum() + 1e-6)

    def _supervised_weight_stats(self, sample_weight: Optional[torch.Tensor]) -> Dict[str, float]:
        out = {
            'priority_is_weighted_loss': 0.0,
            'is_weight_mean': 1.0,
            'is_weight_max': 1.0,
            'is_weight_min': 1.0,
        }
        if sample_weight is None:
            return out
        weight = sample_weight.detach().float().reshape(-1)
        if weight.numel() == 0:
            return out
        out['priority_is_weighted_loss'] = 1.0
        out['is_weight_mean'] = float(weight.mean().item())
        out['is_weight_max'] = float(weight.max().item())
        out['is_weight_min'] = float(weight.min().item())
        return out

    def _compute_uncertainty_metrics_stats(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_log_std: Optional[torch.Tensor],
        prefix: str = 'metrics_',
    ) -> Dict[str, float]:
        """
        Compute monitoring metrics for heteroscedastic Gaussian predictions:
        - NLL (mean and per-dim)
        - Coverage for central credible intervals p in {50, 80, 90, 95}%
        - Calibration error as mean(|emp_cov(p) - p|) over those p values
        """
        # Always return a stable key set so DI-engine monitor_vars doesn't break.
        out: Dict[str, float] = {}
        dim = int(pred.shape[-1])
        out[prefix + 'has_log_std'] = 1.0 if pred_log_std is not None else 0.0
        out[prefix + 'nll'] = 0.0
        out[prefix + 'calib_err'] = 0.0
        for i in range(dim):
            out[prefix + f'nll_dim{i}'] = 0.0
            out[prefix + f'calib_err_dim{i}'] = 0.0
        for p in [50, 80, 90, 95]:
            out[prefix + f'coverage_{p}'] = 0.0
            for i in range(dim):
                out[prefix + f'coverage_{p}_dim{i}'] = 0.0

        if pred_log_std is None:
            return out

        with torch.no_grad():
            pred = pred.detach().reshape(-1, dim)
            target = target.detach().reshape(-1, dim)
            pred_log_std = pred_log_std.detach().reshape(-1, dim)
            # Robust clamp to avoid inf var; calibration may shift log_std.
            pred_log_std = pred_log_std.clamp(min=-10.0, max=10.0)
            diff = pred - target
            var = torch.exp(2.0 * pred_log_std)
            nll = 0.5 * (diff ** 2) / (var + 1e-12) + pred_log_std
            nll_dim = nll.mean(dim=0)
            out[prefix + 'nll'] = float(nll_dim.mean().item())
            for i in range(dim):
                out[prefix + f'nll_dim{i}'] = float(nll_dim[i].item())

            std = torch.exp(pred_log_std)
            # Central interval coverage based on Normal quantiles.
            ps = [0.50, 0.80, 0.90, 0.95]
            norm = torch.distributions.Normal(
                torch.tensor(0.0, device=pred.device, dtype=pred.dtype),
                torch.tensor(1.0, device=pred.device, dtype=pred.dtype),
            )
            covs = []
            covs_dim = []
            for p in ps:
                q = torch.tensor(0.5 + p / 2.0, device=pred.device, dtype=pred.dtype)
                z = norm.icdf(q)
                inside = (diff.abs() <= (z * std)).float()  # (N, D)
                cov_dim = inside.mean(dim=0)
                cov = float(cov_dim.mean().item())
                covs.append(cov)
                covs_dim.append(cov_dim)
                out[prefix + f'coverage_{int(p * 100)}'] = cov
                for i in range(dim):
                    out[prefix + f'coverage_{int(p * 100)}_dim{i}'] = float(cov_dim[i].item())

            # Calibration error: mean absolute gap between empirical coverage and nominal p.
            calib_err = float(np.mean([abs(c - p) for c, p in zip(covs, ps)]))
            out[prefix + 'calib_err'] = calib_err
            for i in range(dim):
                covs_i = [float(cd[i].item()) for cd in covs_dim]
                out[prefix + f'calib_err_dim{i}'] = float(np.mean([abs(c - p) for c, p in zip(covs_i, ps)]))

        return out

    def _compute_metrics_loss_analysis_stats(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_log_std: Optional[torch.Tensor],
        prefix: str = 'metrics_loss_',
    ) -> Dict[str, float]:
        """
        Compute a stable set of diagnostics for the supervised metrics loss so the training
        curve can be decomposed into error scale, uncertainty term, and Huber regularization term.
        """
        cfg = self._cfg.learn
        loss_type = str(getattr(cfg, 'metrics_loss_type', 'mse')).lower()
        beta = float(getattr(cfg, 'metrics_huber_beta', 1.0))
        huber_weight = float(getattr(cfg, 'metrics_huber_weight', 0.1))

        out: Dict[str, float] = {
            prefix + 'abs_err_mean': 0.0,
            prefix + 'abs_err_max': 0.0,
            prefix + 'rmse': 0.0,
            prefix + 'diff_std': 0.0,
            prefix + 'loss_elem_mean': 0.0,
            prefix + 'nll_mean': 0.0,
            prefix + 'huber_mean': 0.0,
            prefix + 'huber_scaled_mean': 0.0,
            prefix + 'huber_quadratic_frac': 0.0,
            prefix + 'std_mean': 0.0,
            prefix + 'var_mean': 0.0,
            prefix + 'beta': beta,
            prefix + 'huber_weight': huber_weight,
            prefix + 'use_nll': 1.0 if loss_type in ['gaussian_nll', 'nll', 'gaussian_nll_huber'] else 0.0,
            prefix + 'use_huber': 1.0 if loss_type in ['huber', 'smooth_l1', 'gaussian_nll_huber'] else 0.0,
        }

        with torch.no_grad():
            diff = (pred - target).detach().float()
            abs_diff = diff.abs()
            out[prefix + 'abs_err_mean'] = float(abs_diff.mean().item())
            out[prefix + 'abs_err_max'] = float(abs_diff.max().item())
            out[prefix + 'rmse'] = float(torch.sqrt((diff ** 2).mean()).item())
            out[prefix + 'diff_std'] = float(diff.std(unbiased=False).item())

            loss_elem = diff ** 2
            nll = None
            huber = None

            if loss_type in ['gaussian_nll', 'nll', 'gaussian_nll_huber'] and pred_log_std is not None:
                pred_log_std = pred_log_std.detach().float().clamp(min=-10.0, max=10.0)
                var = torch.exp(2.0 * pred_log_std)
                std = torch.exp(pred_log_std)
                nll = 0.5 * (diff ** 2) / (var + 1e-12) + pred_log_std
                loss_elem = nll
                out[prefix + 'nll_mean'] = float(nll.mean().item())
                out[prefix + 'std_mean'] = float(std.mean().item())
                out[prefix + 'var_mean'] = float(var.mean().item())

            if loss_type in ['huber', 'smooth_l1', 'gaussian_nll_huber']:
                huber = torch.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)
                out[prefix + 'huber_mean'] = float(huber.mean().item())
                out[prefix + 'huber_scaled_mean'] = float((huber_weight * huber).mean().item())
                out[prefix + 'huber_quadratic_frac'] = float((abs_diff < beta).float().mean().item())
                if loss_type in ['huber', 'smooth_l1']:
                    loss_elem = huber
                elif nll is not None:
                    loss_elem = nll + huber_weight * huber

            out[prefix + 'loss_elem_mean'] = float(loss_elem.mean().item())

        return out

    def _compute_w_aug_loss(
        self,
        pred_metrics_taken: torch.Tensor,
        obs_tensor: torch.Tensor,
    ) -> Tuple[float, Optional[torch.Tensor]]:
        cfg = getattr(self._cfg.learn, 'metrics_w_aug', None)
        if not cfg or not bool(getattr(cfg, 'enable', False)):
            return 0.0, None

        num_samples = int(getattr(cfg, 'num_samples', 4))
        noise_std = float(getattr(cfg, 'noise_std', 0.05))
        weight = float(getattr(cfg, 'loss_weight', 0.1))
        normalize = bool(getattr(cfg, 'normalize_w', True))
        clamp_min = getattr(cfg, 'clamp_min', 0.0)
        clamp_max = getattr(cfg, 'clamp_max', None)
        eps = 1e-6

        w = self._extract_preference_w(obs_tensor).detach()
        u = (pred_metrics_taken * w).sum(dim=-1)
        loss = 0.0
        for _ in range(num_samples):
            w_tilde = w + torch.randn_like(w) * noise_std
            if clamp_min is not None:
                w_tilde = w_tilde.clamp(min=float(clamp_min))
            if clamp_max is not None:
                w_tilde = w_tilde.clamp(max=float(clamp_max))
            if normalize:
                denom = w_tilde.sum(dim=-1, keepdim=True).clamp(min=eps)
                w_tilde = w_tilde / denom
            u_tilde = (pred_metrics_taken * w_tilde).sum(dim=-1)
            loss = loss + F.mse_loss(u_tilde, u)
        loss = loss / max(num_samples, 1)
        return weight * loss, loss

    def _compute_grad_norm(self, norm_type: float = 2.0) -> float:
        norm_type = float(norm_type)
        if norm_type == float('inf'):
            max_norm = 0.0
            for p in self._model.parameters():
                if p.grad is not None:
                    param_norm = float(p.grad.data.abs().max().item())
                    if param_norm > max_norm:
                        max_norm = param_norm
            return max_norm
        total = 0.0
        for p in self._model.parameters():
            if p.grad is not None:
                param_norm = float(p.grad.data.norm(norm_type).item())
                total += param_norm ** norm_type
        return float(total ** (1.0 / norm_type)) if total > 0 else 0.0

    def _data_preprocess_learn(self, data: List[Dict[str, Any]]) -> dict:
        r"""
        Overview:
            Preprocess the data to fit the required data format for learning

        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): the data collected from collect function

        Returns:
            - data (:obj:`Dict[str, Any]`): the processed data, including at least \
                ['main_obs', 'target_obs', 'burnin_obs', 'action', 'reward', 'done', 'weight']
            - data_info (:obj:`dict`): the data info, such as replay_buffer_idx, replay_unique_id
        """
        # data preprocess
        data = timestep_collate(data)
        if self._cuda:
            data = to_device(data, self._device)

        if self._priority_IS_weight:
            assert self._priority, "Use IS Weight correction, but Priority is not used."
        if self._priority and self._priority_IS_weight:
            data['weight'] = data['IS']
        else:
            data['weight'] = data.get('weight', None)

        burnin_step = self._burnin_step
        supervised_metrics_only = bool(getattr(self._cfg.learn, 'supervised_metrics_only', False))

        # In supervised mode, we don't need done/value_gamma/target_obs slicing logic; keep the whole unroll.
        if supervised_metrics_only:
            if 'metrics' not in data:
                raise KeyError(
                    "supervised_metrics_only=True requires `metrics` in training data; "
                    "check env wrapper to attach finalized_sample['metrics']."
                )
            # Keep per-timestep tensors aligned with main_obs.
            data['action'] = data['action'][burnin_step:]
            data['reward'] = data['reward'][burnin_step:]
            data['metrics'] = data['metrics'][burnin_step:]
            data['burnin_obs'] = data['obs'][:burnin_step]
            data['main_obs'] = data['obs'][burnin_step:]
            # Keep optional tensors aligned with main_obs when available.
            if 'done' in data and torch.is_tensor(data['done']):
                data['done'] = data['done'][burnin_step:]
            else:
                data['done'] = data.get('done', None)
            data['value_gamma'] = data.get('value_gamma', None)
            data['weight'] = data.get('weight', None)
            return data

        # data['done'], data['weight'], data['value_gamma'] is used in def _forward_learn() to calculate
        # the q_nstep_td_error, should be length of [self._unroll_len_add_burnin_step-self._burnin_step]
        ignore_done = self._cfg.learn.ignore_done
        if ignore_done:
            data['done'] = [None for _ in range(self._unroll_len_add_burnin_step - burnin_step)]
        else:
            data['done'] = data['done'][burnin_step:].float()  # for computation of online model self._learn_model
            # NOTE that after the proprocessing of  get_nstep_return_data() in _get_train_sample
            # the data['done'] [t] is already the n-step done

        # if the data don't include 'weight' or 'value_gamma' then fill in None in a list
        # with length of [self._unroll_len_add_burnin_step-self._burnin_step],
        # below is two different implementation ways
        if 'value_gamma' not in data:
            data['value_gamma'] = [None for _ in range(self._unroll_len_add_burnin_step - burnin_step)]
        else:
            data['value_gamma'] = data['value_gamma'][burnin_step:]

        if 'weight' not in data or data['weight'] is None:
            data['weight'] = [None for _ in range(self._unroll_len_add_burnin_step - burnin_step)]
        else:
            data['weight'] = data['weight'] * torch.ones_like(data['done'])
            # every timestep in sequence has same weight, which is the _priority_IS_weight in PER

        # cut the seq_len from burn_in step to (seq_len - nstep) step
        data['action'] = data['action'][burnin_step:-self._nstep]
        # cut the seq_len from burn_in step to (seq_len - nstep) step
        data['reward'] = data['reward'][burnin_step:-self._nstep]

        if 'metrics' in data:
            data['metrics'] = data['metrics'][burnin_step:-self._nstep]

        # the burnin_nstep_obs is used to calculate the init hidden state of rnn for the calculation of the q_value,
        # target_q_value, and target_q_action

        # these slicing are all done in the outermost layer, which is the seq_len dim
        data['burnin_nstep_obs'] = data['obs'][:burnin_step + self._nstep]
        # the main_obs is used to calculate the q_value, the [bs:-self._nstep] means using the data from
        # [bs] timestep to [self._unroll_len_add_burnin_step-self._nstep] timestep
        data['main_obs'] = data['obs'][burnin_step:-self._nstep]
        # the target_obs is used to calculate the target_q_value
        data['target_obs'] = data['obs'][burnin_step + self._nstep:]

        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
            Acquire the data, calculate the loss and optimize learner model.

        Arguments:
            - data (:obj:`dict`): Dict type data, including at least \
                ['main_obs', 'target_obs', 'burnin_obs', 'action', 'reward', 'done', 'weight']

        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including cur_lr and total_loss
                - cur_lr (:obj:`float`): Current learning rate
                - total_loss (:obj:`float`): The calculated loss
        """
        # forward
        try:
            data = self._data_preprocess_learn(data)  # output datatype: Dict
        except Exception as e:
            print("Get a exception, data:", data)
            raise e
        supervised_metrics_only = bool(getattr(self._cfg.learn, 'supervised_metrics_only', False))
        self._learn_model.train()
        self._target_model.train()
        # use the hidden state in timestep=0
        # note the reset method is performed at the hidden state wrapper, to reset self._state.
        self._learn_model.reset(data_id=None, state=data['prev_state'][0])
        self._target_model.reset(data_id=None, state=data['prev_state'][0])

        if supervised_metrics_only:
            # Supervised regression on `metrics`. No TD loss, no target model usage.
            if self._burnin_step > 0 and len(data.get('burnin_obs', [])) != 0:
                with torch.no_grad():
                    burnin_out = self._learn_model.forward(
                        {'obs': data['burnin_obs'], 'enable_fast_timestep': True},
                        saved_hidden_state_timesteps=[self._burnin_step],
                    )
                    burnin_out_target = self._target_model.forward(
                        {'obs': data['burnin_obs'], 'enable_fast_timestep': True},
                        saved_hidden_state_timesteps=[self._burnin_step],
                    )
                self._learn_model.reset(data_id=None, state=burnin_out['saved_hidden_state'][0])
                self._target_model.reset(data_id=None, state=burnin_out_target['saved_hidden_state'][0])

            inputs = {'obs': data['main_obs'], 'enable_fast_timestep': True}
            learn_output = self._learn_model.forward(inputs)
            if 'pred_metrics' not in learn_output:
                raise KeyError(
                    "supervised_metrics_only=True requires model output `pred_metrics`. "
                    "Check DiscreteHead/DuelingHead implementation."
                )
            true_metrics = data['metrics']  # expected (T, B, D)
            action = data['action']  # (T, B)
            sample_weight = data.get('weight', None)
            supervised_weight_stats = self._supervised_weight_stats(sample_weight)

            if true_metrics.dim() == 2:
                true_metrics = true_metrics.unsqueeze(0)
            # If ensemble heads exist, supervise each head and average the loss to preserve diversity.
            pred_metrics_log_std_taken_for_stats = None
            if 'pred_metrics_ens' in learn_output:
                # print("Using ensemble heads for metrics prediction and loss calculation.")
                pred_metrics_ens = learn_output['pred_metrics_ens']  # (E,T,B,A,D)其中 T=时间步，B=批次大小，A=动作空间，D=指标维度）
                pred_metrics_log_std_ens = learn_output.get('pred_metrics_log_std_ens', None)
                action_for_metrics = action.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
                    pred_metrics_ens.shape[0], -1, -1, 1, pred_metrics_ens.shape[-1]
                )#(1,T,B,1,1)->(E,T,B,1,D)
                pred_metrics_taken_ens = pred_metrics_ens.gather(3, action_for_metrics).squeeze(3)  # 取出执行action的预测metrics(E,T,B,D)
                if pred_metrics_log_std_ens is not None:
                    pred_metrics_log_std_taken_ens = pred_metrics_log_std_ens.gather(
                        3, action_for_metrics
                    ).squeeze(3)# 取出执行action的预测metrics的log_std(E,T,B,D)
                    pred_metrics_log_std_taken_ens_raw = pred_metrics_log_std_taken_ens
                    self._maybe_calibrate_metrics(
                        pred_metrics_taken_ens.mean(dim=0),
                        pred_metrics_log_std_taken_ens_raw.mean(dim=0),
                        true_metrics,
                    )
                    pred_metrics_log_std_taken_ens = self._apply_metrics_calibration(pred_metrics_log_std_taken_ens_raw)
                    pred_metrics_log_std_taken_for_stats = pred_metrics_log_std_taken_ens.mean(dim=0)#(T,B,D)
                else:
                    pred_metrics_log_std_taken_ens = None
                true_metrics_ens = true_metrics.unsqueeze(0).expand_as(pred_metrics_taken_ens)
                loss_per_sample = self._compute_metrics_loss(
                    pred_metrics_taken_ens.float(),
                    true_metrics_ens.float(),
                    pred_log_std=pred_metrics_log_std_taken_ens,
                    reduction='none',
                )
                bootstrap_prob = float(getattr(self._cfg.learn, 'metrics_bootstrap_prob', 1.0))
                if bootstrap_prob < 1.0:
                    mask = (torch.rand_like(loss_per_sample) < bootstrap_prob).float()
                    metrics_loss = self._reduce_supervised_loss(
                        loss_per_sample,
                        sample_weight=sample_weight,
                        mask=mask,
                    )
                else:
                    metrics_loss = self._reduce_supervised_loss(loss_per_sample, sample_weight=sample_weight)
                pred_metrics_taken = pred_metrics_taken_ens.mean(dim=0)  # (T,B,D), for logging/priority
            else:
                pred_metrics = learn_output['pred_metrics']  # (T, B, A, D)
                pred_metrics_log_std = learn_output.get('pred_metrics_log_std', None)
                action_for_metrics = action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, pred_metrics.shape[-1])
                pred_metrics_taken = pred_metrics.gather(2, action_for_metrics).squeeze(2)  # (T,B,D)
                if pred_metrics_log_std is not None:
                    pred_metrics_log_std_taken_raw = pred_metrics_log_std.gather(2, action_for_metrics).squeeze(2)
                    self._maybe_calibrate_metrics(
                        pred_metrics_taken,
                        pred_metrics_log_std_taken_raw,
                        true_metrics,
                    )
                    pred_metrics_log_std_taken = self._apply_metrics_calibration(pred_metrics_log_std_taken_raw)
                    pred_metrics_log_std_taken_for_stats = pred_metrics_log_std_taken
                else:
                    pred_metrics_log_std_taken = None
                loss_per_sample = self._compute_metrics_loss(
                    pred_metrics_taken.float(),
                    true_metrics.float(),
                    pred_log_std=pred_metrics_log_std_taken,
                    reduction='none',
                )
                metrics_loss_unweighted = loss_per_sample.mean()
                metrics_loss = self._reduce_supervised_loss(loss_per_sample, sample_weight=sample_weight)
            if 'metrics_loss_unweighted' not in locals():
                metrics_loss_unweighted = loss_per_sample.mean()
            metrics_loss_weight = float(getattr(self._cfg.learn, 'metrics_loss_weight', 1.0))
            loss = metrics_loss_weight * metrics_loss

            obs_tensor = self._extract_obs_tensor(data['main_obs'])
            w_aug_loss, w_aug_loss_raw = self._compute_w_aug_loss(pred_metrics_taken, obs_tensor)
            loss = loss + w_aug_loss

            # Priority: per-sample mixture of max/mean abs error over time.
            per_timestep_abs = (pred_metrics_taken.detach() - true_metrics.detach()).abs().mean(dim=-1)  # (T,B)
            td_error_per_sample = 0.9 * per_timestep_abs.max(dim=0)[0] + 0.1 * per_timestep_abs.mean(dim=0)

            self._optimizer.zero_grad()
            loss.backward()
            grad_norm = self._compute_grad_norm(getattr(self._cfg.learn, 'grad_clip_norm_type', 2.0))
            self._optimizer.step()
            self._target_model.update(self._learn_model.state_dict())

            stats = self._compute_uncertainty_metrics_stats(
                pred_metrics_taken.float(),
                true_metrics.float(),
                pred_metrics_log_std_taken_for_stats.float() if pred_metrics_log_std_taken_for_stats is not None else None,
                prefix='metrics_',
            )
            loss_stats = self._compute_metrics_loss_analysis_stats(
                pred_metrics_taken.float(),
                true_metrics.float(),
                pred_metrics_log_std_taken_for_stats.float() if pred_metrics_log_std_taken_for_stats is not None else None,
                prefix='metrics_loss_',
            )

            # Diagnostics for monitor vars (compute meaningful values in supervised mode).
            q_value = learn_output.get('logit', None)
            batch_range = torch.arange(action.shape[1], device=action.device)
            q_s_a_t0 = None
            q_s_a_mean_t0 = 0.0
            if torch.is_tensor(q_value):
                q_s_a_t0 = q_value[0][batch_range, action[0]]
                q_s_a_mean_t0 = q_value[0].mean().item()

            target_q_s_a_t0 = None
            with torch.no_grad():
                target_out = self._target_model.forward({'obs': data['main_obs'], 'enable_fast_timestep': True})
                target_q_value = target_out.get('logit', None)
                if torch.is_tensor(target_q_value):
                    if 'action' in learn_output:
                        target_q_action = learn_output['action']
                    elif torch.is_tensor(q_value):
                        target_q_action = q_value.argmax(dim=-1)
                    else:
                        target_q_action = action
                    target_q_s_a_t0 = target_q_value[0][batch_range, target_q_action[0]]

            # Utility signals for taken action.
            obs_tensor = self._extract_obs_tensor(data['main_obs'])
            u_pred_t0 = None
            u_true_t0 = None
            u_abs_err_t0 = None
            if torch.is_tensor(obs_tensor):
                w = self._extract_preference_w(obs_tensor)
                u_pred = (pred_metrics_taken * w).sum(dim=-1)
                u_true = (true_metrics * w).sum(dim=-1)
                u_pred_t0 = u_pred[0]
                u_true_t0 = u_true[0]
                u_abs_err_t0 = (u_pred_t0 - u_true_t0).abs()

            # RL-style return diagnostics from sampled rewards.
            nstep_return_t0 = torch.zeros(action.shape[1], device=action.device, dtype=pred_metrics_taken.dtype)
            reward = data.get('reward', None)
            if torch.is_tensor(reward):
                if reward.dim() == 2:
                    nstep_return_t0 = reward[0].float()
                    nstep = 1
                elif reward.dim() == 3:
                    # Support both (T, B, nstep) and (T, nstep, B).
                    if reward.shape[1] == action.shape[1]:
                        reward_t = reward.permute(0, 2, 1).contiguous()
                    else:
                        reward_t = reward
                    nstep = reward_t.shape[1]
                    gammas = (self._gamma ** torch.arange(nstep, device=reward_t.device, dtype=reward_t.dtype)).view(1, nstep, 1)
                    nstep_return_t0 = (reward_t[0:1] * gammas).sum(dim=1).squeeze(0)
                else:
                    nstep = self._nstep
            else:
                nstep = self._nstep

            done = data.get('done', None)
            if done is None:
                not_done = torch.ones_like(nstep_return_t0)
            elif torch.is_tensor(done):
                if done.dim() == 0:
                    not_done = (1.0 - done.float()).repeat(nstep_return_t0.shape[0])
                elif done.dim() == 1:
                    not_done = (1.0 - done.float()).view(-1)
                else:
                    not_done = (1.0 - done[0].float()).view(-1)
            else:
                not_done = torch.ones_like(nstep_return_t0)

            if target_q_s_a_t0 is not None:
                target_return_t0 = nstep_return_t0 + (self._gamma ** nstep) * not_done * target_q_s_a_t0.detach()
            else:
                target_return_t0 = nstep_return_t0

            # Uncertainty signals.
            epistemic_std_taken_t0 = 0.0
            if 'logit_std' in learn_output and torch.is_tensor(learn_output['logit_std']):
                logit_std_t0 = learn_output['logit_std'][0]
                epistemic_std_taken_t0 = float(logit_std_t0[batch_range, action[0]].mean().item())

            aleatoric_sigma_u_taken_t0 = 0.0
            if pred_metrics_log_std_taken_for_stats is not None and torch.is_tensor(obs_tensor):
                w_t0 = self._extract_preference_w(obs_tensor)[0]
                log_std_t0 = torch.clamp(pred_metrics_log_std_taken_for_stats[0], min=-10.0, max=10.0)
                std_t0 = torch.exp(log_std_t0)
                sigma_u_sq = ((w_t0 * std_t0) ** 2).sum(dim=-1)
                aleatoric_sigma_u_taken_t0 = float(torch.sqrt(sigma_u_sq + 1e-8).mean().item())

            calib_log_t_scalars = self._calibration_scalar_dict()

            return {
                'cur_lr': self._optimizer.defaults['lr'],
                'grad_norm': grad_norm,
                'total_loss': loss.item(),
                'metrics_loss': metrics_loss.item(),
                'metrics_loss_unweighted': float(metrics_loss_unweighted.item()),
                'is_weighted_metrics_loss': metrics_loss.item(),
                'weighted_minus_unweighted_metrics_loss': float(metrics_loss.item() - metrics_loss_unweighted.item()),
                'metrics_w_aug_loss': 0.0 if w_aug_loss_raw is None else float(w_aug_loss_raw.item()),
                'priority': td_error_per_sample.tolist(),
                **supervised_weight_stats,
                **stats,
                **loss_stats,
                **calib_log_t_scalars,
                'q_s_taken-a_t0': 0.0 if q_s_a_t0 is None else q_s_a_t0.mean().item(),
                'target_q_s_max-a_t0': 0.0 if target_q_s_a_t0 is None else target_q_s_a_t0.mean().item(),
                'q_s_a-mean_t0': q_s_a_mean_t0,
                'nstep_return_t0': nstep_return_t0.mean().item(),
                'target_return_t0': target_return_t0.mean().item(),
                'q_minus_target_return_t0': 0.0 if q_s_a_t0 is None else (q_s_a_t0.detach() - target_return_t0).mean().item(),
                'u_pred_t0': 0.0 if u_pred_t0 is None else u_pred_t0.mean().item(),
                'u_true_t0': 0.0 if u_true_t0 is None else u_true_t0.mean().item(),
                'u_abs_err_t0': 0.0 if u_abs_err_t0 is None else u_abs_err_t0.mean().item(),
                'epistemic_std_taken_t0': epistemic_std_taken_t0,
                'aleatoric_sigma_u_taken_t0': aleatoric_sigma_u_taken_t0,
            }

        if len(data['burnin_nstep_obs']) != 0:
            with torch.no_grad():
                inputs = {'obs': data['burnin_nstep_obs'], 'enable_fast_timestep': True}
                burnin_output = self._learn_model.forward(
                    inputs, saved_hidden_state_timesteps=[self._burnin_step, self._burnin_step + self._nstep]
                )  # keys include 'logit', 'hidden_state' 'saved_hidden_state', \
                # 'action', for their specific dim, please refer to DRQN model
                burnin_output_target = self._target_model.forward(
                    inputs, saved_hidden_state_timesteps=[self._burnin_step, self._burnin_step + self._nstep]
                )

        self._learn_model.reset(data_id=None, state=burnin_output['saved_hidden_state'][0])
        inputs = {'obs': data['main_obs'], 'enable_fast_timestep': True}
        if 'metrics' in data:
            inputs['metrics'] = data['metrics']
        learn_output = self._learn_model.forward(inputs)
        q_value = learn_output['logit']
        self._learn_model.reset(data_id=None, state=burnin_output['saved_hidden_state'][1])
        self._target_model.reset(data_id=None, state=burnin_output_target['saved_hidden_state'][1])

        next_inputs = {'obs': data['target_obs'], 'enable_fast_timestep': True}
        with torch.no_grad():
            target_q_value = self._target_model.forward(next_inputs)['logit']
            # argmax_action double_dqn
            target_q_action = self._learn_model.forward(next_inputs)['action']

        action, reward, done, weight = data['action'], data['reward'], data['done'], data['weight']
        value_gamma = data['value_gamma']
        # Handle case when nstep=1: reward might be (T, B) instead of (T, B, nstep)
        if reward.dim() == 2:
            # Add nstep dimension: (T, B) -> (T, B, 1)
            reward = reward.unsqueeze(-1)
        # T, B, nstep -> T, nstep, B
        reward = reward.permute(0, 2, 1).contiguous()
        loss = []
        td_error = []
        for t in range(self._unroll_len_add_burnin_step - self._burnin_step - self._nstep):
            # here t=0 means timestep <self._burnin_step> in the original sample sequence, we minus self._nstep
            # because for the last <self._nstep> timestep in the sequence, we don't have their target obs
            td_data = q_nstep_td_data(
                q_value[t], target_q_value[t], action[t], target_q_action[t], reward[t], done[t], weight[t]
            )
            if self._value_rescale:
                l, e = q_nstep_td_error_with_rescale(td_data, self._gamma, self._nstep, value_gamma=value_gamma[t])
                loss.append(l)
                td_error.append(e.abs())
            else:
                l, e = q_nstep_td_error(td_data, self._gamma, self._nstep, value_gamma=value_gamma[t])
                loss.append(l)
                # td will be a list of the length (self._unroll_len_add_burnin_step - self._burnin_step - self._nstep)
                # and each value is a tensor of the size batch_size
                td_error.append(e.abs())
        loss = sum(loss) / (len(loss) + 1e-8)

        # compute metrics MSE loss if present
        metrics_loss_val = 0.0
        w_aug_loss_raw = None
        stats = self._compute_uncertainty_metrics_stats(
            torch.zeros((1, 1, self._metrics_dim), device=q_value.device, dtype=q_value.dtype),
            torch.zeros((1, 1, self._metrics_dim), device=q_value.device, dtype=q_value.dtype),
            None,
            prefix='metrics_',
        )
        loss_stats = self._compute_metrics_loss_analysis_stats(
            torch.zeros((1, 1, self._metrics_dim), device=q_value.device, dtype=q_value.dtype),
            torch.zeros((1, 1, self._metrics_dim), device=q_value.device, dtype=q_value.dtype),
            None,
            prefix='metrics_loss_',
        )
        if 'metrics' in data and 'pred_metrics' in learn_output:
            true_metrics = data['metrics']  # (T, B, D)
            pred_metrics = learn_output['pred_metrics']  # (T, B, A, D)
            T, B = action.shape
            
            if true_metrics.dim() == 2:
                true_metrics = true_metrics.unsqueeze(0) # handle shape mismatch
            
            # expand action for gathering (T, B, 1, D)
            action_for_metrics = action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, pred_metrics.shape[-1])
            # get the predicted metrics for the taken action
            pred_metrics_taken = pred_metrics.gather(2, action_for_metrics).squeeze(2)

            pred_metrics_log_std = learn_output.get('pred_metrics_log_std', None)
            if pred_metrics_log_std is not None:
                pred_metrics_log_std_taken_raw = pred_metrics_log_std.gather(2, action_for_metrics).squeeze(2)
                self._maybe_calibrate_metrics(
                    pred_metrics_taken,
                    pred_metrics_log_std_taken_raw,
                    true_metrics,
                )
                pred_metrics_log_std_taken = self._apply_metrics_calibration(pred_metrics_log_std_taken_raw)
                stats = self._compute_uncertainty_metrics_stats(
                    pred_metrics_taken.float(),
                    true_metrics.float(),
                    pred_metrics_log_std_taken.float(),
                    prefix='metrics_',
                )
            else:
                pred_metrics_log_std_taken = None

            loss_stats = self._compute_metrics_loss_analysis_stats(
                pred_metrics_taken.float(),
                true_metrics.float(),
                pred_metrics_log_std_taken.float() if pred_metrics_log_std_taken is not None else None,
                prefix='metrics_loss_',
            )

            metrics_loss = self._compute_metrics_loss(
                pred_metrics_taken.float(),
                true_metrics.float(),
                pred_log_std=pred_metrics_log_std_taken,
            )
            loss = loss + metrics_loss
            metrics_loss_val = metrics_loss.item()

            obs_tensor = self._extract_obs_tensor(data['main_obs'])
            w_aug_loss, w_aug_loss_raw = self._compute_w_aug_loss(pred_metrics_taken, obs_tensor)
            loss = loss + w_aug_loss

        # using the mixture of max and mean absolute n-step TD-errors as the priority of the sequence
        td_error_per_sample = 0.9 * torch.max(
            torch.stack(td_error), dim=0
        )[0] + (1 - 0.9) * (torch.sum(torch.stack(td_error), dim=0) / (len(td_error) + 1e-8))
        # torch.max(torch.stack(td_error), dim=0) will return tuple like thing, please refer to torch.max
        # td_error shape list(<self._unroll_len_add_burnin_step-self._burnin_step-self._nstep>, B), for example, (75,64)
        # torch.sum(torch.stack(td_error), dim=0) can also be replaced with sum(td_error)

        # update
        self._optimizer.zero_grad()
        loss.backward()
        grad_norm = self._compute_grad_norm(getattr(self._cfg.learn, 'grad_clip_norm_type', 2.0))
        self._optimizer.step()
        # after update
        self._target_model.update(self._learn_model.state_dict())

        # the information for debug
        batch_range = torch.arange(action[0].shape[0])
        q_s_a_t0 = q_value[0][batch_range, action[0]]
        target_q_s_a_t0 = target_q_value[0][batch_range, target_q_action[0]]

        # Extra debug/visualization signals:
        # - "nstep_return_t0": discounted sum of the sampled n-step rewards (no bootstrap).
        # - "target_return_t0": the TD target used by Double-DQN (with bootstrap, masking by done).
        # These help visualize q(s,a) vs "realized" return/target.
        with torch.no_grad():
            # reward is (T, nstep, B)
            nstep = reward.shape[1]
            gammas = (self._gamma ** torch.arange(nstep, device=reward.device, dtype=reward.dtype)).view(1, nstep, 1)
            nstep_return_t0 = (reward[0:1] * gammas).sum(dim=1).squeeze(0)  # (B,)
            if done[0] is None:
                not_done = torch.ones_like(nstep_return_t0)
            else:
                not_done = (1.0 - done[0].float()).view(-1)
            target_return_t0 = nstep_return_t0 + (self._gamma ** self._nstep) * not_done * target_q_s_a_t0.detach()

        # Utility (w · metrics) for the taken action vs "true" metrics label.
        u_pred_t0 = None
        u_true_t0 = None
        u_abs_err_t0 = None
        with torch.no_grad():
            if 'metrics' in data and 'pred_metrics' in learn_output:
                true_metrics = data['metrics']  # (T, B, D)
                pred_metrics = learn_output['pred_metrics']  # (T, B, A, D)
                if true_metrics.dim() == 2:
                    true_metrics = true_metrics.unsqueeze(0)
                action_for_metrics = action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, pred_metrics.shape[-1])
                pred_metrics_taken = pred_metrics.gather(2, action_for_metrics).squeeze(2)  # (T, B, D)
                obs_tensor = self._extract_obs_tensor(data['main_obs'])
                if torch.is_tensor(obs_tensor):
                    w = self._extract_preference_w(obs_tensor)  # (T, B, D)
                    u_pred = (pred_metrics_taken * w).sum(dim=-1)  # (T, B)
                    u_true = (true_metrics * w).sum(dim=-1)  # (T, B)
                    u_pred_t0 = u_pred[0]
                    u_true_t0 = u_true[0]
                    u_abs_err_t0 = (u_pred_t0 - u_true_t0).abs()

        # Uncertainty signals (if the model supports them).
        epistemic_std_taken_t0 = None
        aleatoric_sigma_u_taken_t0 = None
        with torch.no_grad():
            if 'logit_std' in learn_output:
                logit_std_t0 = learn_output['logit_std'][0]  # (B, A)
                epistemic_std_taken_t0 = logit_std_t0[batch_range, action[0]].mean().item()
            # Aleatoric uncertainty over utility u = w·m.
            if 'pred_metrics_log_std' in learn_output:
                obs_tensor = self._extract_obs_tensor(data['main_obs'])
                if torch.is_tensor(obs_tensor):
                    w = self._extract_preference_w(obs_tensor)[0]  # (B, D)
                    aleatoric_log_std_t0 = learn_output['pred_metrics_log_std'][0]  # (B, A, D)
                    aleatoric_log_std_t0 = self._apply_metrics_calibration(aleatoric_log_std_t0)
                    aleatoric_log_std_t0 = torch.clamp(aleatoric_log_std_t0, min=-10.0, max=10.0)
                    aleatoric_std_t0 = torch.exp(aleatoric_log_std_t0)  # (B, A, D)
                    sigma_u_sq = ((w.unsqueeze(1) * aleatoric_std_t0) ** 2).sum(dim=-1)  # (B, A)
                    sigma_u = torch.sqrt(sigma_u_sq + 1e-8)  # (B, A)
                    aleatoric_sigma_u_taken_t0 = sigma_u[batch_range, action[0]].mean().item()

        calib_log_t_scalars = self._calibration_scalar_dict()

        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'grad_norm': grad_norm,
            'total_loss': loss.item(),
            'metrics_loss': metrics_loss_val,
            'metrics_loss_unweighted': metrics_loss_val,
            'is_weighted_metrics_loss': metrics_loss_val,
            'weighted_minus_unweighted_metrics_loss': 0.0,
            'metrics_w_aug_loss': 0.0 if w_aug_loss_raw is None else float(w_aug_loss_raw.item()),
            'priority': td_error_per_sample.tolist(),  # note abs operation has been performed above
            'priority_is_weighted_loss': 0.0,
            'is_weight_mean': 1.0,
            'is_weight_max': 1.0,
            'is_weight_min': 1.0,
            **stats,
            **loss_stats,
            # Keep the raw list for debugging, but don't rely on it for TB scalars.
            **calib_log_t_scalars,
            # the first timestep in the sequence, may not be the start of episode
            'q_s_taken-a_t0': q_s_a_t0.mean().item(),
            'target_q_s_max-a_t0': target_q_s_a_t0.mean().item(),
            'q_s_a-mean_t0': q_value[0].mean().item(),
            'nstep_return_t0': nstep_return_t0.mean().item(),
            'target_return_t0': target_return_t0.mean().item(),
            'q_minus_target_return_t0': (q_s_a_t0.detach() - target_return_t0).mean().item(),
            'u_pred_t0': 0.0 if u_pred_t0 is None else u_pred_t0.mean().item(),
            'u_true_t0': 0.0 if u_true_t0 is None else u_true_t0.mean().item(),
            'u_abs_err_t0': 0.0 if u_abs_err_t0 is None else u_abs_err_t0.mean().item(),
            'epistemic_std_taken_t0': 0.0 if epistemic_std_taken_t0 is None else float(epistemic_std_taken_t0),
            'aleatoric_sigma_u_taken_t0': 0.0 if aleatoric_sigma_u_taken_t0 is None else float(aleatoric_sigma_u_taken_t0),
            '[histogram]q_s_taken_t0': q_s_a_t0.detach().cpu(),
            '[histogram]target_return_t0': target_return_t0.detach().cpu(),
            '[histogram]q_minus_target_return_t0': (q_s_a_t0.detach() - target_return_t0).detach().cpu(),
            '[histogram]u_abs_err_t0': (torch.zeros_like(q_s_a_t0) if u_abs_err_t0 is None else u_abs_err_t0).detach().cpu(),
        }

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        self._learn_model.reset(data_id=data_id)

    def _state_dict_learn(self) -> Dict[str, Any]:
        state = {
            'model': self._learn_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }
        if self._metrics_log_t is not None:
            state['metrics_log_t'] = self._metrics_log_t.detach().cpu()
            state['metrics_calibration_step'] = int(self._metrics_calibration_step)
        return state

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])
        if self._metrics_log_t is not None and 'metrics_log_t' in state_dict:
            self._metrics_log_t.copy_(state_dict['metrics_log_t'].to(self._metrics_log_t.device))
            self._metrics_calibration_step = int(state_dict.get('metrics_calibration_step', 0))

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
        """
        self._metrics_dim = self._get_metrics_dim()
        assert 'unroll_len' not in self._cfg.collect, "r2d2 use default unroll_len"
        self._nstep = self._cfg.nstep
        self._burnin_step = self._cfg.burnin_step
        self._gamma = self._cfg.discount_factor
        self._unroll_len_add_burnin_step = self._cfg.unroll_len + self._cfg.burnin_step
        self._unroll_len = self._unroll_len_add_burnin_step  # for compatibility

        # for r2d2, this hidden_state wrapper is to add the 'prev hidden state' for each transition.
        # Note that collect env forms a batch and the key is added for the batch simultaneously.
        self._collect_model = model_wrap(
            self._model, wrapper_name='hidden_state', state_num=self._cfg.collect.env_num * self._cfg.collect.max_agent_num , save_prev_state=True
        )
        self._collect_model.reset()

    def _forward_collect(self, data: dict, eps: float, data_id: List[int] = None) -> dict:
        r"""
        Overview:
            Forward function for collect mode with eps_greedy
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
            - eps (:obj:`float`): epsilon value for exploration, which is decayed by collected env step.
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.
        ReturnsKeys
            - necessary: ``action``
        """
        if data_id is None:
            data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        obs = data
        exploration_type = getattr(self._cfg.collect, 'exploration_type', 'eps_greedy')
        ucb_beta = float(getattr(self._cfg.collect, 'ucb_beta', 1.0))

        # 1) Decide action using the current hidden state, but do not mutate hidden state during MC sampling.
        # HiddenStateWrapper maintains state in self._collect_model._state (a dict keyed by data_id).
        prev_state = [self._collect_model._state[i] for i in data_id]
        model_inp = {'obs': obs, 'prev_state': prev_state}

        self._collect_model.eval()
        base_model = self._collect_model._model  # unwrap HiddenStateWrapper
        with torch.no_grad():
            out = base_model.forward(model_inp, inference=True)
            mean = out['logit']
            # 1. 提取 Epistemic Uncertainty (认知不确定性，用于探索)
            epistemic_std = out.get('logit_std', torch.zeros_like(mean))

            # 2. 提取 Aleatoric Uncertainty (偶然不确定性，用于风险回避)
            # 注意：这需要你在 q_learning.py 的 _ensemble_head_num == 1 或多头分支中，
            # 确保 inference=True 时把 'pred_metrics_log_std' 返回到 out 字典里
            aleatoric_log_std = out.get('pred_metrics_log_std', None)

            # 安全地提取 w，防止 obs 是 dict 导致崩溃
            obs_tensor = self._extract_obs_tensor(obs)
            w = self._extract_preference_w(obs_tensor)  # (B, D)

            # 3. 正确合成效用的 Aleatoric 抖动 (sigma_u)
            if aleatoric_log_std is not None:
                # 假设各 metric 之间独立: Var(u) = sum( w_i^2 * Var(m_i) )
                aleatoric_log_std = self._apply_metrics_calibration(aleatoric_log_std)
                # Avoid overflow if calibration pushes values out of head clamp range.
                aleatoric_log_std = torch.clamp(aleatoric_log_std, min=-10.0, max=10.0)
                aleatoric_std = torch.exp(aleatoric_log_std)  # (B, A, D)
                w_expanded = w.unsqueeze(1)  # (B, 1, D)
                # 计算方差和
                sigma_u_sq = ((w_expanded * aleatoric_std) ** 2).sum(dim=-1)  # (B, A)
                # 开方得到效用 u 的标准差，加 1e-8 防止数值下溢导致 nan (虽然这里是 no_grad)
                sigma_u = torch.sqrt(sigma_u_sq + 1e-8)
            else:
                sigma_u = torch.zeros_like(mean)

            # 获取超参数
            risk_alpha = float(getattr(self._cfg.collect, 'risk_alpha', 0.0)) # 风险厌恶惩罚系数

            # 4. 计算最终 Score (UCB 探索未知 + 惩罚高风险)
            if exploration_type == 'ucb':
                # mean + 乐观探索 - 风险回避
                score = mean + ucb_beta * epistemic_std - risk_alpha * sigma_u
            elif exploration_type == 'thompson':
                # 修复 std 未定义: 从后验分布 (mean, epistemic_std) 中采样
                score = mean + epistemic_std * torch.randn_like(epistemic_std) 
                # 同样施加环境固有抖动的惩罚
                score = score - risk_alpha * sigma_u
            else: # eps_greedy 或其他
                score = mean - risk_alpha * sigma_u

            greedy_action = score.argmax(dim=-1)

            action_dim = int(mean.shape[-1])
            # Epsilon mixture on top of greedy/UCB action. This keeps behavior policy simple and records propensity.
            rand = torch.rand_like(greedy_action.float())
            random_action = torch.randint(0, action_dim, greedy_action.shape, device=greedy_action.device)
            take_random = rand < float(eps)
            action = torch.where(take_random, random_action, greedy_action).long()
            # Propensity is exact for eps-greedy / UCB+eps. For Thompson, this is only an approximation.
            propensity = (float(eps) / action_dim) + (1.0 - float(eps)) * (action == greedy_action).float()

            # 2) Advance hidden state exactly once via wrapper forward.
            output = self._collect_model.forward({'obs': obs}, data_id=data_id, inference=True)
            output['action'] = action
            output['propensity'] = propensity
            if 'logit_std' in out:
                output['logit_std'] = out['logit_std']
            # Extra per-step visualization signals (store only "taken" scalars to reduce log volume).
            batch_range = torch.arange(action.shape[0], device=action.device)
            output['q_value_taken'] = mean[batch_range, action]  # (B,)
            output['q_value_max'] = mean.max(dim=-1)[0]  # (B,)
            output['ucb_score_taken'] = score[batch_range, action]  # (B,)
            output['ucb_score_max'] = score.max(dim=-1)[0]  # (B,)
            output['epistemic_std_taken'] = epistemic_std[batch_range, action]  # (B,)
            output['aleatoric_sigma_u_taken'] = sigma_u[batch_range, action]  # (B,)

            # If model predicts metrics, also log taken prediction and aleatoric std per metric.
            if 'pred_metrics' in out:
                pm = out['pred_metrics']  # (B, A, D)
                output['pred_metrics_taken'] = pm[batch_range, action]
                if 'pred_metrics_log_std' in out and out['pred_metrics_log_std'] is not None:
                    pms = self._apply_metrics_calibration(out['pred_metrics_log_std'])
                    pms = torch.clamp(pms, min=-10.0, max=10.0)
                    output['pred_metrics_std_taken'] = torch.exp(pms)[batch_range, action]
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        self._collect_model.reset(data_id=data_id)

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action', 'prev_state']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'action': model_output['action'],
            'prev_state': model_output['prev_state'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        if 'propensity' in model_output:
            transition['propensity'] = model_output['propensity']
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and the n step return data, then sample from the n_step return data

        Arguments:
            - data (:obj:`list`): The trajectory's cache

        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        for i in range(len(data)):
            data[i]["action"] = data[i]["action"].long()
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len_add_burnin_step, overlap = self._cfg.unroll_overlap)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy.
        """
        self._metrics_dim = self._get_metrics_dim()
        self._eval_model = model_wrap(self._model, wrapper_name='hidden_state', state_num=self._cfg.eval.env_num * self._cfg.eval.max_agent_num)
        self._eval_model = model_wrap(self._eval_model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: dict, data_id: List[int] = None) -> dict:
        r"""
        Overview:
            Forward function of eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, data_id=data_id, inference=True)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        self._eval_model.reset(data_id=data_id)

    def default_model(self) -> Tuple[str, List[str]]:
        return 'drqn', ['ding.model.template.q_learning']

    def _monitor_vars_learn(self) -> List[str]:
        vars = super()._monitor_vars_learn() + [
            'grad_norm',
            'total_loss',
            'metrics_loss',
            'metrics_loss_unweighted',
            'is_weighted_metrics_loss',
            'weighted_minus_unweighted_metrics_loss',
            'metrics_w_aug_loss',
            'priority',
            'priority_is_weighted_loss',
            'is_weight_mean',
            'is_weight_max',
            'is_weight_min',
            'metrics_loss_abs_err_mean',
            'metrics_loss_abs_err_max',
            'metrics_loss_rmse',
            'metrics_loss_diff_std',
            'metrics_loss_loss_elem_mean',
            'metrics_loss_nll_mean',
            'metrics_loss_huber_mean',
            'metrics_loss_huber_scaled_mean',
            'metrics_loss_huber_quadratic_frac',
            'metrics_loss_std_mean',
            'metrics_loss_var_mean',
            'metrics_loss_beta',
            'metrics_loss_huber_weight',
            'metrics_loss_use_nll',
            'metrics_loss_use_huber',
            # Heteroscedastic uncertainty monitoring
            'metrics_has_log_std',
            'metrics_nll',
            'metrics_coverage_50',
            'metrics_coverage_80',
            'metrics_coverage_90',
            'metrics_coverage_95',
            'metrics_calib_err',
            'q_s_taken-a_t0',
            'target_q_s_max-a_t0',
            'q_s_a-mean_t0',
            'nstep_return_t0',
            'target_return_t0',
            'q_minus_target_return_t0',
            'u_pred_t0',
            'u_true_t0',
            'u_abs_err_t0',
            'epistemic_std_taken_t0',
            'aleatoric_sigma_u_taken_t0',
        ]
        for i in range(self._metrics_dim):
            vars.append(f'metrics_nll_dim{i}')
            vars.append(f'metrics_calib_log_t_{i}')
            vars.append(f'metrics_calib_t_{i}')
        return vars
