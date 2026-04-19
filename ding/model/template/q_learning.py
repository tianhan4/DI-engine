from typing import Union, Optional, Dict, Callable, List
import torch
import torch.nn as nn

from ding.torch_utils import get_lstm
from ding.utils import MODEL_REGISTRY, SequenceType, squeeze
from ..common import FCEncoder, ConvEncoder, DiscreteHead, DuelingHead, MultiHead, RainbowHead, \
    QuantileHead, QRDQNHead, DistributionHead


@MODEL_REGISTRY.register('dqn')
class DQN(nn.Module):

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            dueling: bool = True,
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        """
        Overview:
            Init the DQN (encoder + head) Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape, such as 6 or [2, 3, 3].
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
                the last element must match ``head_hidden_size``.
            - dueling (:obj:`dueling`): Whether choose ``DuelingHead`` or ``DiscreteHead(default)``.
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of head network.
            - head_layer_num (:obj:`int`): The number of layers used in the head network to compute Q value output
            - activation (:obj:`Optional[nn.Module]`): The type of activation function in networks \
                if ``None`` then default set it to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see \
                ``ding.torch_utils.fc_block`` for more details.
        """
        super(DQN, self).__init__()
        # For compatibility: 1, (1, ), [4, 32, 32]
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
        if head_hidden_size is None:
            head_hidden_size = encoder_hidden_size_list[-1]
        # FC Encoder
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = FCEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        # Conv Encoder
        elif len(obs_shape) == 3:
            self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".format(obs_shape)
            )
        # Head Type
        if dueling:
            head_cls = DuelingHead
        else:
            head_cls = DiscreteHead
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self.head = MultiHead(
                head_cls,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                activation=activation,
                norm_type=norm_type
            )
        else:
            self.head = head_cls(
                head_hidden_size, action_shape, head_layer_num, activation=activation, norm_type=norm_type
            )

    def forward(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            DQN forward computation graph, input observation tensor to predict q_value.
        Arguments:
            - x (:obj:`torch.Tensor`): Observation inputs
        Returns:
            - outputs (:obj:`Dict`): DQN forward outputs, such as q_value.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Discrete Q-value output of each action dimension.
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is ``obs_shape``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, M)`, where B is batch size and M is ``action_shape``
        Examples:
            >>> model = DQN(32, 6)  # arguments: 'obs_shape' and 'action_shape'
            >>> inputs = torch.randn(4, 32)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict) and outputs['logit'].shape == torch.Size([4, 6])
        """
        x = self.encoder(x)
        x = self.head(x)
        return x


@MODEL_REGISTRY.register('c51dqn')
class C51DQN(nn.Module):

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        encoder_hidden_size_list: SequenceType = [128, 128, 64],
        head_hidden_size: int = None,
        head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        v_min: Optional[float] = -10,
        v_max: Optional[float] = 10,
        n_atom: Optional[int] = 51,
    ) -> None:
        r"""
        Overview:
            Init the C51 Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to ``Head``.
            - head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
            - n_atom (:obj:`Optional[int]`): Number of atoms in the prediction distribution.
        """
        super(C51DQN, self).__init__()
        # For compatibility: 1, (1, ), [4, 32, 32]
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
        if head_hidden_size is None:
            head_hidden_size = encoder_hidden_size_list[-1]
        # FC Encoder
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = FCEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        # Conv Encoder
        elif len(obs_shape) == 3:
            self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own C51DQN".format(obs_shape)
            )
        # Head Type
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self.head = MultiHead(
                DistributionHead,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                activation=activation,
                norm_type=norm_type,
                n_atom=n_atom,
                v_min=v_min,
                v_max=v_max,
            )
        else:
            self.head = DistributionHead(
                head_hidden_size,
                action_shape,
                head_layer_num,
                activation=activation,
                norm_type=norm_type,
                n_atom=n_atom,
                v_min=v_min,
                v_max=v_max,
            )

    def forward(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            Use observation tensor to predict C51DQN's output.
            Parameter updates with C51DQN's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor w/ ``(B, N=head_hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head. Return the result prediction dictionary.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``x``.
            - distribution (:obj:`torch.Tensor`): Distribution tensor of size ``(B, N, n_atom)``
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is head_hidden_size.
            - logit (:obj:`torch.FloatTensor`): :math:`(B, M)`, where M is action_shape.
            - distribution(:obj:`torch.FloatTensor`): :math:`(B, M, P)`, where P is n_atom.

        Examples:
            >>> model = C51DQN(128, 64)  # arguments: 'obs_shape' and 'action_shape'
            >>> inputs = torch.randn(4, 128)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict)
            >>> # default head_hidden_size: int = 64,
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
            >>> # default n_atom: int = 51
            >>> assert outputs['distribution'].shape == torch.Size([4, 64, 51])
        """
        x = self.encoder(x)
        x = self.head(x)
        return x


@MODEL_REGISTRY.register('qrdqn')
class QRDQN(nn.Module):

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            num_quantiles: int = 32,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
    ) -> None:
        r"""
        Overview:
            Init the QRDQN Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to ``Head``.
            - head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output
            - num_quantiles (:obj:`int`): Number of quantiles in the prediction distribution.
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
        """
        super(QRDQN, self).__init__()
        # For compatibility: 1, (1, ), [4, 32, 32]
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
        if head_hidden_size is None:
            head_hidden_size = encoder_hidden_size_list[-1]
        # FC Encoder
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = FCEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        # Conv Encoder
        elif len(obs_shape) == 3:
            self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own QRDQN".format(obs_shape)
            )
        # Head Type
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self.head = MultiHead(
                QRDQNHead,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                num_quantiles=num_quantiles,
                activation=activation,
                norm_type=norm_type,
            )
        else:
            self.head = QRDQNHead(
                head_hidden_size,
                action_shape,
                head_layer_num,
                num_quantiles=num_quantiles,
                activation=activation,
                norm_type=norm_type,
            )

    def forward(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            Use observation tensor to predict QRDQN's output.
            Parameter updates with QRDQN's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor with ``(B, N=hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head. Return the result prediction dictionary.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``x``.
            - q (:obj:`torch.Tensor`): Q valye tensor tensor of size ``(B, N, num_quantiles)``
            - tau (:obj:`torch.Tensor`): tau tensor of size ``(B, N, 1)``
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is head_hidden_size.
            - logit (:obj:`torch.FloatTensor`): :math:`(B, M)`, where M is action_shape.
            - tau (:obj:`torch.Tensor`):  :math:`(B, M, 1)`

        Examples:
            >>> model = QRDQN(64, 64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
            >>> # default num_quantiles : int = 32
            >>> assert outputs['q'].shape == torch.Size([4, 64, 32])
            >>> assert outputs['tau'].shape == torch.Size([4, 32, 1])
        """
        x = self.encoder(x)
        x = self.head(x)
        return x


@MODEL_REGISTRY.register('iqn')
class IQN(nn.Module):

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            num_quantiles: int = 32,
            quantile_embedding_size: int = 128,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        r"""
        Overview:
            Init the IQN Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape.
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to ``Head``.
            - head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output
            - num_quantiles (:obj:`int`): Number of quantiles in the prediction distribution.
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details.
        """
        super(IQN, self).__init__()
        # For compatibility: 1, (1, ), [4, 32, 32]
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
        if head_hidden_size is None:
            head_hidden_size = encoder_hidden_size_list[-1]
        # FC Encoder
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = FCEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        # Conv Encoder
        elif len(obs_shape) == 3:
            self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own IQN".format(obs_shape)
            )
        # Head Type
        head_cls = QuantileHead
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self.head = MultiHead(
                head_cls,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                num_quantiles=num_quantiles,
                quantile_embedding_size=quantile_embedding_size,
                activation=activation,
                norm_type=norm_type
            )
        else:
            self.head = head_cls(
                head_hidden_size,
                action_shape,
                head_layer_num,
                activation=activation,
                norm_type=norm_type,
                num_quantiles=num_quantiles,
                quantile_embedding_size=quantile_embedding_size,
            )

    def forward(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            Use encoded embedding tensor to predict IQN's output.
            Parameter updates with IQN's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor with ``(B, N=hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head. Return the result prediction dictionary.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``x``.
            - q (:obj:`torch.Tensor`): Q valye tensor tensor of size ``(num_quantiles, N, B)``
            - quantiles (:obj:`torch.Tensor`): quantiles tensor of size ``(quantile_embedding_size, 1)``
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is head_hidden_size.
            - logit (:obj:`torch.FloatTensor`): :math:`(B, M)`, where M is action_shape
            - quantiles (:obj:`torch.Tensor`):  :math:`(P, 1)`, where P is quantile_embedding_size.
        Examples:
            >>> model = IQN(64, 64) # arguments: 'obs_shape' and 'action_shape'
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
            >>> # default num_quantiles: int = 32
            >>> assert outputs['q'].shape == torch.Size([32, 4, 64]
            >>> # default quantile_embedding_size: int = 128
            >>> assert outputs['quantiles'].shape == torch.Size([128, 1])
        """
        x = self.encoder(x)
        x = self.head(x)
        return x


@MODEL_REGISTRY.register('rainbowdqn')
class RainbowDQN(nn.Module):
    """
    Overview:
        RainbowDQN network (C51 + Dueling + Noisy Block)

    .. note::
        RainbowDQN contains dueling architecture by default
    """

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        encoder_hidden_size_list: SequenceType = [128, 128, 64],
        head_hidden_size: Optional[int] = None,
        head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        v_min: Optional[float] = -10,
        v_max: Optional[float] = 10,
        n_atom: Optional[int] = 51,
    ) -> None:
        """
        Overview:
            Init the Rainbow Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape.
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to ``Head``.
            - head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output
            - activation (:obj:`Optional[nn.Module]`): The type of activation function to use in ``MLP`` the after \
                ``layer_fn``, if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`): The type of normalization to use, see ``ding.torch_utils.fc_block`` \
                for more details`
            - n_atom (:obj:`Optional[int]`): Number of atoms in the prediction distribution.
        """
        super(RainbowDQN, self).__init__()
        # For compatibility: 1, (1, ), [4, 32, 32]
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
        if head_hidden_size is None:
            head_hidden_size = encoder_hidden_size_list[-1]
        # FC Encoder
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = FCEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        # Conv Encoder
        elif len(obs_shape) == 3:
            self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own RainbowDQN".
                format(obs_shape)
            )
        # Head Type
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self.head = MultiHead(
                RainbowHead,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                activation=activation,
                norm_type=norm_type,
                n_atom=n_atom,
                v_min=v_min,
                v_max=v_max,
            )
        else:
            self.head = RainbowHead(
                head_hidden_size,
                action_shape,
                head_layer_num,
                activation=activation,
                norm_type=norm_type,
                n_atom=n_atom,
                v_min=v_min,
                v_max=v_max,
            )

    def forward(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            Use observation tensor to predict Rainbow output.
            Parameter updates with Rainbow's MLPs forward setup.
        Arguments:
            - x (:obj:`torch.Tensor`):
                The encoded embedding tensor with ``(B, N=hidden_size)``.
        Returns:
            - outputs (:obj:`Dict`):
                Run ``MLP`` with ``RainbowHead`` setups and return the result prediction dictionary.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``x``.
            - distribution (:obj:`torch.Tensor`): Distribution tensor of size ``(B, N, n_atom)``
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is head_hidden_size.
            - logit (:obj:`torch.FloatTensor`): :math:`(B, M)`, where M is action_shape.
            - distribution(:obj:`torch.FloatTensor`): :math:`(B, M, P)`, where P is n_atom.

        Examples:
            >>> model = RainbowDQN(64, 64) # arguments: 'obs_shape' and 'action_shape'
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])
            >>> # default n_atom: int =51
            >>> assert outputs['distribution'].shape == torch.Size([4, 64, 51])
        """
        x = self.encoder(x)
        x = self.head(x)
        return x


def myparallel_wrapper(forward_fn: Callable) -> Callable:
    r"""
    Overview:
        Process timestep T and batch_size B at the same time, in other words, treat different timestep data as
        different trajectories in a batch.
    Arguments:
        - forward_fn (:obj:`Callable`): Normal ``nn.Module`` 's forward function.
    Returns:
        - wrapper (:obj:`Callable`): Wrapped function.
    """

    def wrapper(x: torch.Tensor, w: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        T, B = x.shape[:2]
        # print("x,shape: ", x.shape)
        # print("w.shape: ", w.shape)
        def reshape(d):
            if isinstance(d, list):
                d = [reshape(t) for t in d]
            elif isinstance(d, dict):
                d = {k: reshape(v) for k, v in d.items()}
            else:
                d = d.reshape(T, B, *d.shape[1:])
            return d

        x = x.reshape(T * B, *x.shape[2:])
        w = w.reshape(T * B, *w.shape[2:])
        x = forward_fn(x, w)
        x = reshape(x)
        return x

    return wrapper

def parallel_wrapper(forward_fn: Callable) -> Callable:
    r"""
    Overview:
        Process timestep T and batch_size B at the same time, in other words, treat different timestep data as
        different trajectories in a batch.
    Arguments:
        - forward_fn (:obj:`Callable`): Normal ``nn.Module`` 's forward function.
    Returns:
        - wrapper (:obj:`Callable`): Wrapped function.
    """

    def wrapper(x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        T, B = x.shape[:2]

        def reshape(d):
            if isinstance(d, list):
                d = [reshape(t) for t in d]
            elif isinstance(d, dict):
                d = {k: reshape(v) for k, v in d.items()}
            else:
                d = d.reshape(T, B, *d.shape[1:])
            return d

        x = x.reshape(T * B, *x.shape[2:])
        x = forward_fn(x)
        x = reshape(x)
        return x

    return wrapper


@MODEL_REGISTRY.register('drqn')
class DRQN(nn.Module):
    """
    Overview:
        DQN + RNN = DRQN
    """

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            dueling: bool = False,
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            lstm_type: Optional[str] = 'normal',
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            res_link: bool = False,
            ensemble_head_num: int = 1,
            metrics_dim: int = 4,
            action_conditioned_metrics_head: bool = False,
            action_embed_dim: Optional[int] = None,
            gaussian_metrics_head: bool = False,
            metrics_log_std_min: float = -5.0,
            metrics_log_std_max: float = 2.0,
            metrics_sigma_eps: float = 1e-8,
    ) -> None:
        r"""
        Overview:
            Init the DRQN Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to ``Head``.
            - lstm_type (:obj:`Optional[str]`): Version of rnn cell, now support ['normal', 'pytorch', 'hpc', 'gru']
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
            - res_link (:obj:`bool`): use the residual link or not, default to False
        """
        super(DRQN, self).__init__()
        # For compatibility: 1, (1, ), [4, 32, 32]
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
        self._metrics_dim = int(metrics_dim)
        if self._metrics_dim <= 0:
            raise ValueError(f"metrics_dim must be positive, got {self._metrics_dim}")
        # This repo assumes the last metrics_dim dims of obs are preference vector w.
        if isinstance(obs_shape, int):
            if obs_shape <= self._metrics_dim:
                raise ValueError(
                    f"obs_shape must be > {self._metrics_dim} to include state + {self._metrics_dim}-dim preference w, got {obs_shape}"
                )
            encoder_obs_shape = obs_shape - self._metrics_dim
        elif isinstance(obs_shape, (tuple, list)) and len(obs_shape) == 1:
            if obs_shape[0] <= self._metrics_dim:
                raise ValueError(
                    f"obs_shape[0] must be > {self._metrics_dim} to include state + {self._metrics_dim}-dim preference w, got {obs_shape[0]}"
                )
            encoder_obs_shape = [obs_shape[0] - self._metrics_dim]
        else:
            raise RuntimeError(
                f"DRQN in this repo expects 1D obs with last {self._metrics_dim} dims as preference w, got obs_shape={obs_shape}"
            )
        if head_hidden_size is None:
            head_hidden_size = encoder_hidden_size_list[-1]
        # FC Encoder
        if isinstance(encoder_obs_shape, int) or len(encoder_obs_shape) == 1:
            self.encoder = FCEncoder(
                encoder_obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
            )
        # LSTM Type
        self.rnn = get_lstm(lstm_type, input_size=head_hidden_size, hidden_size=head_hidden_size)
        self.res_link = res_link
        # Head Type
        if dueling:
            head_cls = DuelingHead
        else:
            head_cls = DiscreteHead
        multi_head = not isinstance(action_shape, int)
        head_kwargs = dict(layer_num=head_layer_num, activation=activation, norm_type=norm_type)
        head_kwargs['metrics_dim'] = self._metrics_dim
        head_kwargs['action_conditioned_metrics'] = bool(action_conditioned_metrics_head)
        head_kwargs['action_embed_dim'] = action_embed_dim
        head_kwargs['gaussian_metrics'] = bool(gaussian_metrics_head)
        head_kwargs['log_std_min'] = metrics_log_std_min
        head_kwargs['log_std_max'] = metrics_log_std_max
        head_kwargs['sigma_eps'] = metrics_sigma_eps

        if multi_head:
            # Not used in this repo's 3-action discrete bandit setup.
            raise NotImplementedError(
                "MultiHead with preference-weighted Discrete/Dueling heads is not supported in this repo."
            )
        else:
            self._ensemble_head_num = int(ensemble_head_num) # 集成头数
            if self._ensemble_head_num < 1: # 如果集成头数小于1，则抛出错误
                raise ValueError(f"ensemble_head_num must be  = 1, got {self._ensemble_head_num}")
            if self._ensemble_head_num == 1: # 如果集成头数为1，则使用单个头
                self.head = head_cls(head_hidden_size, action_shape, **head_kwargs)
            else: # 如果集成头数大于1，则使用多个头
                # Shared encoder + shared RNN, multiple independent heads for epistemic uncertainty.
                self.heads = nn.ModuleList([head_cls(head_hidden_size, action_shape, **head_kwargs)
                                            for _ in range(self._ensemble_head_num)])
                # 创建一个包含多个头的模块列表，每个头都使用相同的参数配置。
                # 这样，在训练过程中，每个头可以独立地进行预测，从而产生不同的预测结果。
                # 这对于不确定性估计（如 MC-dropout）特别有用，因为它允许模型在不同的时间步或不同的数据子集上进行预测，并计算这些预测结果的方差。
                # 通过这种方式，可以更准确地估计不确定性，并提供更可靠的决策支持。
    def forward(
            self, inputs: Dict, inference: bool = False, saved_hidden_state_timesteps: Optional[list] = None
    ) -> Dict:
        r"""
        Overview:
            Use observation tensor to predict DRQN output.
            Parameter updates with DRQN's MLPs forward setup.
        Arguments:
            - inputs (:obj:`Dict`):
            - inference: (:obj:'bool'): if inference is True, we unroll the one timestep transition,
                if inference is False, we unroll the sequence transitions.
            - saved_hidden_state_timesteps: (:obj:'Optional[list]'): when inference is False,
                we unroll the sequence transitions, then we would save rnn hidden states at timesteps
                that are listed in list saved_hidden_state_timesteps.

       ArgumentsKeys:
            - obs (:obj:`torch.Tensor`): Encoded observation
            - prev_state (:obj:`list`): Previous state's tensor of size ``(B, N)``

        Returns:
            - outputs (:obj:`Dict`):
                Run ``MLP`` with ``DRQN`` setups and return the result prediction dictionary.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``obs``.
            - next_state (:obj:`list`): Next state's tensor of size ``(B, N)``
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N=obs_space)`, where B is batch size.
            - prev_state(:obj:`torch.FloatTensor list`): :math:`[(B, N)]`
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`
            - next_state(:obj:`torch.FloatTensor list`): :math:`[(B, N)]`

        Examples:
            >>> # Init input's Keys:
            >>> prev_state = [[torch.randn(1, 1, 64) for __ in range(2)] for _ in range(4)] # B=4
            >>> obs = torch.randn(4,64)
            >>> model = DRQN(64, 64) # arguments: 'obs_shape' and 'action_shape'
            >>> outputs = model({'obs': inputs, 'prev_state': prev_state}, inference=True)
            >>> # Check outputs's Keys
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == (4, 64)
            >>> assert len(outputs['next_state']) == 4
            >>> assert all([len(t) == 2 for t in outputs['next_state']])
            >>> assert all([t[0].shape == (1, 1, 64) for t in outputs['next_state']])
        """

        x, prev_state = inputs['obs'], inputs['prev_state']
        metrics = inputs.get('metrics', None)

        # print("x.shape: ", x.shape)
        # print("x : ", x)
        # for both inference and other cases, the network structure is encoder -> rnn network -> head
        # the difference is inference take the data with seq_len=1 (or T = 1)
        # print("q_learninglwb")
        # print("x.shape: ", x.shape)
        if inference:
            if x.shape[-1] < self._metrics_dim:
                raise RuntimeError(
                    f"DRQN inference expects obs with last {self._metrics_dim} dims as preference w, got shape={tuple(x.shape)}"
                )
            w = x[:, -self._metrics_dim:]
            x = x[:, :-self._metrics_dim]
            x = self.encoder(x)  # 现在输入是5维，匹配 encoder 的 in_features
            if self.res_link:
                a = x
            x = x.unsqueeze(0)  # for rnn input, put the seq_len of x as 1 instead of none.
            # prev_state: DataType: List[Tuple[torch.Tensor]]; Initially, it is a list of None
            x, next_state = self.rnn(x, prev_state)
            x = x.squeeze(0)  # to delete the seq_len dim to match head network input
            if self.res_link:
                x = x + a
            if getattr(self, '_ensemble_head_num', 1) == 1:
                out = self.head(x, w)
                out['next_state'] = next_state
                return out
            else:
                logits = []
                pred_metrics = []
                pred_metrics_log_std = []
                aleatoric_sigma_u = []
                for head in self.heads:
                    h_out = head(x, w)
                    logits.append(h_out['logit'])
                    if 'pred_metrics' in h_out:
                        pred_metrics.append(h_out['pred_metrics'])
                    if 'pred_metrics_log_std' in h_out:
                        pred_metrics_log_std.append(h_out['pred_metrics_log_std'])
                    if 'aleatoric_sigma_u' in h_out:
                        aleatoric_sigma_u.append(h_out['aleatoric_sigma_u'])
                logits = torch.stack(logits, dim=0)  # (E,B,A)
                out = {
                    'logit': logits.mean(dim=0),
                    'logit_std': logits.std(dim=0, unbiased=False),
                    'logit_ens': logits,
                    'next_state': next_state,
                }
                if len(pred_metrics) == len(self.heads):
                    pm = torch.stack(pred_metrics, dim=0)  # (E,B,A,D)
                    out.update({
                        'pred_metrics': pm.mean(dim=0), 
                        'pred_metrics_std': pm.std(dim=0, unbiased=False),
                        'pred_metrics_ens': pm, 
                    })
                if len(pred_metrics_log_std) == len(self.heads):
                    pls = torch.stack(pred_metrics_log_std, dim=0)  # (E,B,A,D)
                    out.update({
                        'pred_metrics_log_std': pls.mean(dim=0),
                        'pred_metrics_log_std_ens': pls,
                    })
                if len(aleatoric_sigma_u) == len(self.heads):
                    asu = torch.stack(aleatoric_sigma_u, dim=0)  # (E,B,A)
                    out.update({
                        'aleatoric_sigma_u': asu.mean(dim=0),
                        'aleatoric_sigma_u_std': asu.std(dim=0, unbiased=False),
                        'aleatoric_sigma_u_ens': asu,
                    })
                return out
        else:
            assert len(x.shape) in [3, 5], x.shape
            if x.shape[-1] < self._metrics_dim:
                raise RuntimeError(
                    f"DRQN training expects obs with last {self._metrics_dim} dims as preference w, got shape={tuple(x.shape)}"
                )
            w = x[:, :, -self._metrics_dim:]
            x = x[:, :, :-self._metrics_dim]
            x = parallel_wrapper(self.encoder)(x)  # 输入就变成了 (T, B, 5)，匹配 encoder
            if self.res_link:
                a = x
            lstm_embedding = []
            # TODO(nyz) how to deal with hidden_size key-value
            hidden_state_list = []
            if saved_hidden_state_timesteps is not None:
                saved_hidden_state = []
            for t in range(x.shape[0]):  # T timesteps
                output, prev_state = self.rnn(x[t:t + 1], prev_state)  # output: (1,B, head_hidden_size)
                if saved_hidden_state_timesteps is not None and t + 1 in saved_hidden_state_timesteps:
                    saved_hidden_state.append(prev_state)
                lstm_embedding.append(output)
                hidden_state = list(zip(*prev_state))  # {list: 2{tuple: B{Tensor:(1, 1, head_hidden_size}}}
                # only keep ht, {list: x.shape[0]{Tensor:(1, batch_size, head_hidden_size)}}
                hidden_state_list.append(torch.cat(hidden_state[0], dim=1))
            x = torch.cat(lstm_embedding, 0)  # (T, B, head_hidden_size)
            if self.res_link:
                x = x + a
            if getattr(self, '_ensemble_head_num', 1) == 1:
                x = myparallel_wrapper(self.head)(x, w)  # (T, B, action_shape)
            else:
                T, B = x.shape[:2]
                x_flat = x.reshape(T * B, *x.shape[2:])
                w_flat = w.reshape(T * B, *w.shape[2:])
                logits = []
                pred_metrics = []
                pred_metrics_log_std = []
                aleatoric_sigma_u = []
                for head in self.heads:
                    h_out = head(x_flat, w_flat)
                    logits.append(h_out['logit'].reshape(T, B, *h_out['logit'].shape[1:]))
                    if 'pred_metrics' in h_out:
                        pred_metrics.append(h_out['pred_metrics'].reshape(T, B, *h_out['pred_metrics'].shape[1:]))
                    if 'pred_metrics_log_std' in h_out:
                        pred_metrics_log_std.append(
                            h_out['pred_metrics_log_std'].reshape(T, B, *h_out['pred_metrics_log_std'].shape[1:])
                        )
                    if 'aleatoric_sigma_u' in h_out:
                        aleatoric_sigma_u.append(
                            h_out['aleatoric_sigma_u'].reshape(T, B, *h_out['aleatoric_sigma_u'].shape[1:])
                        )
                logits = torch.stack(logits, dim=0)  # (E,T,B,A)
                x = {
                    'logit': logits.mean(dim=0),
                    'logit_std': logits.std(dim=0, unbiased=False),
                    'logit_ens': logits,
                }
                if len(pred_metrics) == len(self.heads):
                    pm = torch.stack(pred_metrics, dim=0)  # (E,T,B,A,D)
                    x.update({
                        'pred_metrics': pm.mean(dim=0), #预测的均值的均值
                        'pred_metrics_std': pm.std(dim=0, unbiased=False), #预测的均值的方差
                        'pred_metrics_ens': pm, 
                    })
                if len(pred_metrics_log_std) == len(self.heads):
                    pls = torch.stack(pred_metrics_log_std, dim=0)  # (E,T,B,A,D)
                    x.update({
                        'pred_metrics_log_std': pls.mean(dim=0),
                        'pred_metrics_log_std_ens': pls,
                    })
                if len(aleatoric_sigma_u) == len(self.heads):
                    asu = torch.stack(aleatoric_sigma_u, dim=0)  # (E,T,B,A)
                    x.update({
                        'aleatoric_sigma_u': asu.mean(dim=0),
                        'aleatoric_sigma_u_std': asu.std(dim=0, unbiased=False),
                        'aleatoric_sigma_u_ens': asu,
                    })
            # the last timestep state including h and c for lstm, {list: B{tuple: 2{Tensor:(1, 1, head_hidden_size}}}
            x['next_state'] = prev_state
            # all hidden state h, this returns a tensor of the dim: seq_len*batch_size*head_hidden_size
            # This key is used in qtran, the algorithm requires to retain all h_{t} during training
            x['hidden_state'] = torch.cat(hidden_state_list, dim=-3)
            if saved_hidden_state_timesteps is not None:
                x['saved_hidden_state'] = saved_hidden_state  # the selected saved hidden states, including h and c
            return x


class GeneralQNetwork(nn.Module):
    pass
