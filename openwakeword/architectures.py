# coding=utf-8
"""
Tập hợp các kiến trúc mạng Neural (Neural Network Architectures) bằng PyTorch
cho openWakeWord.

Input convention (tất cả các model đều nhận):
    x: Tensor shape (batch, 16, 96) — 16 frames, 96 embedding features

Các model có sẵn:
  - DNNModel     : Fully-Connected Network (DNN) gốc của openWakeWord
  - RNNModel     : Bidirectional LSTM gốc của openWakeWord
  - TCResNet     : Temporal Convolution ResNet
                   (port từ kws_streaming/tc_resnet.py)
  - DSTCResNet   : Depthwise-Separable TC-ResNet (MatchboxNet)
                   (port từ kws_streaming/ds_tc_resnet.py)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_activation(n_classes: int) -> nn.Module:
    """Trả về Sigmoid (binary) hoặc ReLU (multi-class)."""
    return nn.Sigmoid() if n_classes == 1 else nn.ReLU()


# ---------------------------------------------------------------------------
# 1. DNNModel  (giữ nguyên 100% cấu trúc cũ từ train.py)
# ---------------------------------------------------------------------------

class _FCNBlock(nn.Module):
    """Một khối Fully-Connected chuẩn hóa có trong DNN gốc."""

    def __init__(self, layer_dim: int) -> None:
        super().__init__()
        self.fcn_layer = nn.Linear(layer_dim, layer_dim)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(layer_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.layer_norm(self.fcn_layer(x)))


class DNNModel(nn.Module):
    """Fully-Connected DNN — kiến trúc gốc của openWakeWord.

    Giữ nguyên hoàn toàn so với class ``Net`` lồng trong train.py cũ.
    Input : (batch, 16, 96)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        input_shape: tuple = (16, 96),
        layer_dim: int = 128,
        n_blocks: int = 1,
        n_classes: int = 1,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(input_shape[0] * input_shape[1], layer_dim)
        self.relu1 = nn.ReLU()
        self.layernorm1 = nn.LayerNorm(layer_dim)
        self.blocks = nn.ModuleList([_FCNBlock(layer_dim) for _ in range(n_blocks)])
        self.last_layer = nn.Linear(layer_dim, n_classes)
        self.last_act = _make_activation(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.layernorm1(self.layer1(self.flatten(x))))
        for block in self.blocks:
            x = block(x)
        return self.last_act(self.last_layer(x))


# ---------------------------------------------------------------------------
# 2. RNNModel  (giữ nguyên 100% cấu trúc cũ từ train.py)
# ---------------------------------------------------------------------------

class RNNModel(nn.Module):
    """Bidirectional 2-layer LSTM — kiến trúc gốc của openWakeWord.

    Giữ nguyên hoàn toàn so với class ``Net`` lồng trong train.py cũ.
    Input : (batch, 16, 96)
    Output: (batch, n_classes)
    """

    def __init__(self, input_shape: tuple = (16, 96), n_classes: int = 1) -> None:
        super().__init__()
        self.layer1 = nn.LSTM(
            input_shape[-1], 64,
            num_layers=2, bidirectional=True,
            batch_first=True, dropout=0.0,
        )
        self.layer2 = nn.Linear(64 * 2, n_classes)
        self.layer3 = _make_activation(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.layer1(x)
        return self.layer3(self.layer2(out[:, -1]))


# ---------------------------------------------------------------------------
# 3. TCResNet  (port từ kws_streaming/tc_resnet.py — PyTorch)
#    Paper: https://arxiv.org/pdf/1904.03814.pdf
# ---------------------------------------------------------------------------

class _TCResBlock(nn.Module):
    """Một residual block của TC-ResNet.

    Nếu số filter thay đổi, shortcut sẽ dùng Conv1d 1x1 + BN để match shape.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2  # 'same' padding

        # --- Nhánh chính ---
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(p=dropout)

        # --- Nhánh shortcut ---
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu_out(out + residual)
        out = self.dropout(out)
        return out


class TCResNet(nn.Module):
    """Temporal Convolution ResNet (TC-ResNet).

    Port từ kws_streaming/tc_resnet.py sang PyTorch.
    Paper: https://arxiv.org/pdf/1904.03814.pdf

    Input : (batch, 16, 96) — (batch, time, feature)
    Output: (batch, n_classes)

    Lưu ý về trục:
        PyTorch Conv1d nhận (batch, channels, length).
        Ta coi chiều ``feature`` (96) là channels và ``time`` (16) là length,
        nên ta permute từ (batch, time, feature) -> (batch, feature, time)
        ngay đầu forward.
    """

    def __init__(
        self,
        input_shape: tuple = (16, 96),
        n_classes: int = 1,
        channels: List[int] = None,
        kernel_size: int = 9,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [24, 36, 36, 48, 48, 72, 72]

        in_features = input_shape[1]  # 96

        # --- Lớp Conv đầu tiên (kernel nhỏ, bắt cạnh cứng) ---
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
        )

        # --- Các Residual Block ---
        res_channels = channels[1:]
        blocks: List[nn.Module] = []
        prev_ch = channels[0]
        for out_ch in res_channels:
            stride = 2 if out_ch != prev_ch else 1
            blocks.append(_TCResBlock(prev_ch, out_ch, kernel_size, stride, dropout))
            prev_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # --- Global Average Pool + Dropout + Classifier ---
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(prev_ch, n_classes)
        self.last_act = _make_activation(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time=16, feature=96)
        x = x.permute(0, 2, 1)          # -> (batch, 96, 16)
        x = self.stem(x)                 # -> (batch, C0, T)
        x = self.blocks(x)               # -> (batch, Cn, T')
        x = self.pool(x).squeeze(-1)     # -> (batch, Cn)
        x = self.dropout(x)
        return self.last_act(self.classifier(x))


# ---------------------------------------------------------------------------
# 4. DSTCResNet / MatchboxNet  (port từ kws_streaming/ds_tc_resnet.py)
#    Paper: https://arxiv.org/pdf/2004.08531.pdf
# ---------------------------------------------------------------------------

class _CausalPad(nn.Module):
    """Pad zeros vào đầu chuỗi thời gian (causal padding)."""

    def __init__(self, pad: int) -> None:
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(x, (self.pad, 0))


class _DSBlock(nn.Module):
    """Một Residual Block dùng Depthwise-Separable Conv 1D.

    Cấu trúc mỗi lần lặp bên trong:
        DepthwiseConv1d (ksize) -> Pointwise 1x1 -> BN -> Act -> Dropout
    Cuối block:
        DepthwiseConv1d (ksize) -> Pointwise 1x1 -> BN
        -> (+ shortcut nếu residual=True) -> Act -> Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        repeat: int = 1,
        use_residual: bool = False,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU}
        act_fn = act_map.get(activation, nn.ReLU)

        pad = (kernel_size // 2) * dilation  # 'same' padding

        sub_blocks: List[nn.Module] = []
        ch_in = in_channels
        for i in range(repeat):
            is_last = i == repeat - 1
            # Depthwise Conv (groups=ch_in -> pure depthwise)
            sub_blocks.append(
                nn.Conv1d(
                    ch_in, ch_in, kernel_size,
                    stride=stride if i == 0 else 1,
                    dilation=dilation,
                    padding=pad,
                    groups=ch_in, bias=False,
                )
            )
            # Pointwise 1x1
            sub_blocks.append(nn.Conv1d(ch_in, out_channels, 1, bias=False))
            ch_in = out_channels

            sub_blocks.append(nn.BatchNorm1d(out_channels))
            if not is_last:
                sub_blocks.append(act_fn())
                if dropout > 0:
                    sub_blocks.append(nn.Dropout(p=dropout))

        self.sub_blocks = nn.Sequential(*sub_blocks)

        # Shortcut
        self.use_residual = use_residual
        if use_residual:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = None

        self.act_out = act_fn()
        self.drop_out = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.sub_blocks(x)
        if self.use_residual and self.shortcut is not None:
            out = out + self.shortcut(x)
        return self.drop_out(self.act_out(out))


class DSTCResNet(nn.Module):
    """Depthwise-Separable TC-ResNet (MatchboxNet).

    Port từ kws_streaming/ds_tc_resnet.py sang PyTorch.
    Paper: https://arxiv.org/pdf/2004.08531.pdf

    Input : (batch, 16, 96) — (batch, time, feature)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        input_shape: tuple = (16, 96),
        n_classes: int = 1,
        ds_filters: List[int] = None,
        ds_kernel_size: List[int] = None,
        ds_repeat: List[int] = None,
        ds_dilation: List[int] = None,
        ds_residual: List[int] = None,
        ds_stride: List[int] = None,
        dropout: float = 0.0,
        activation: str = "relu",
        **kwargs,
    ) -> None:
        super().__init__()

        # Giá trị mặc định (từ tham số gốc trong ds_tc_resnet.py)
        if ds_filters is None:
            ds_filters = [128, 64, 64, 64, 128, 128]
        if ds_kernel_size is None:
            ds_kernel_size = [11, 13, 15, 17, 29, 1]
        if ds_repeat is None:
            ds_repeat = [1, 1, 1, 1, 1, 1]
        if ds_dilation is None:
            ds_dilation = [1, 1, 1, 1, 2, 1]
        if ds_residual is None:
            ds_residual = [0, 1, 1, 1, 0, 0]
        if ds_stride is None:
            ds_stride = [1, 1, 1, 1, 1, 1]

        in_ch = input_shape[1]  # 96 — coi feature là channels
        blocks: List[nn.Module] = []
        for filters, ksize, rep, dil, res, stride in zip(
            ds_filters, ds_kernel_size, ds_repeat,
            ds_dilation, ds_residual, ds_stride,
        ):
            blocks.append(
                _DSBlock(
                    in_channels=in_ch,
                    out_channels=filters,
                    kernel_size=ksize,
                    stride=stride,
                    dilation=dil,
                    repeat=rep,
                    use_residual=bool(res),
                    dropout=dropout,
                    activation=activation,
                )
            )
            in_ch = filters

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_ch, n_classes)
        self.last_act = _make_activation(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time=16, feature=96)
        x = x.permute(0, 2, 1)          # -> (batch, 96, 16)
        x = self.blocks(x)               # -> (batch, Cn, T')
        x = self.pool(x).squeeze(-1)     # -> (batch, Cn)
        return self.last_act(self.classifier(x))


# ---------------------------------------------------------------------------
# 5. CNNModel  (port từ kws_streaming/cnn.py)
#    Paper: https://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
# ---------------------------------------------------------------------------

class _Conv1dBnAct(nn.Module):
    """Conv1d -> BatchNorm1d -> Activation — khối cơ bản dùng chung."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1,
                 padding: int = 0, activation: str = "relu") -> None:
        super().__init__()
        act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU}
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              stride=stride, dilation=dilation,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = act_map.get(activation, nn.ReLU)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CNNModel(nn.Module):
    """CNN with dilated convolutions for keyword spotting.

    Port từ kws_streaming/cnn.py sang PyTorch.
    Paper: Convolutional Neural Networks for Small-footprint Keyword Spotting
    https://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

    Input : (batch, 16, 96)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        input_shape: tuple = (16, 96),
        n_classes: int = 1,
        cnn_filters: List[int] = None,
        cnn_kernel_sizes: List[int] = None,
        cnn_dilations: List[int] = None,
        dropout: float = 0.5,
        fc_units: List[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if cnn_filters is None:
            cnn_filters = [64, 64, 64, 64, 128, 64, 128]
        if cnn_kernel_sizes is None:
            cnn_kernel_sizes = [3, 5, 5, 5, 5, 5, 10]
        if cnn_dilations is None:
            cnn_dilations = [1, 1, 2, 1, 2, 1, 2]
        if fc_units is None:
            fc_units = [128, 256]

        in_ch = input_shape[1]  # 96, treat as channels
        conv_layers: List[nn.Module] = []
        for out_ch, ksize, dil in zip(cnn_filters, cnn_kernel_sizes, cnn_dilations):
            pad = (ksize // 2) * dil  # 'same' padding
            conv_layers.append(_Conv1dBnAct(in_ch, out_ch, ksize, dilation=dil, padding=pad))
            in_ch = out_ch
        self.convs = nn.Sequential(*conv_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)

        fc_layers: List[nn.Module] = []
        for units in fc_units:
            fc_layers += [nn.Linear(in_ch, units), nn.ReLU()]
            in_ch = units
        self.fc = nn.Sequential(*fc_layers)

        self.classifier = nn.Linear(in_ch, n_classes)
        self.last_act = _make_activation(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)          # -> (batch, 96, 16)
        x = self.convs(x)               # -> (batch, Cn, T)
        x = self.pool(x).squeeze(-1)    # -> (batch, Cn)
        x = self.dropout(x)
        x = self.fc(x)
        return self.last_act(self.classifier(x))


# ---------------------------------------------------------------------------
# 6. CRNNModel  (port từ kws_streaming/crnn.py)
#    Paper: https://arxiv.org/pdf/1703.05390.pdf
# ---------------------------------------------------------------------------

class CRNNModel(nn.Module):
    """Convolutional Recurrent Neural Network (CRNN) for keyword spotting.

    Port từ kws_streaming/crnn.py sang PyTorch.
    Paper: Convolutional Recurrent Neural Networks for Small-Footprint Keyword Spotting
    https://arxiv.org/pdf/1703.05390.pdf

    Kiến trúc: vài lớp CNN -> GRU -> FC
    Input : (batch, 16, 96)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        input_shape: tuple = (16, 96),
        n_classes: int = 1,
        cnn_filters: List[int] = None,
        cnn_kernel_sizes: List[int] = None,
        gru_units: int = 256,
        dropout: float = 0.1,
        fc_units: List[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if cnn_filters is None:
            cnn_filters = [16, 16]
        if cnn_kernel_sizes is None:
            cnn_kernel_sizes = [3, 5]
        if fc_units is None:
            fc_units = [128, 256]

        in_ch = input_shape[1]  # 96
        conv_layers: List[nn.Module] = []
        for out_ch, ksize in zip(cnn_filters, cnn_kernel_sizes):
            pad = ksize // 2
            conv_layers.append(_Conv1dBnAct(in_ch, out_ch, ksize, padding=pad))
            in_ch = out_ch
        self.convs = nn.Sequential(*conv_layers)

        # GRU nhận (batch, time, features=in_ch)
        self.gru = nn.GRU(in_ch, gru_units, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)

        fc_layers: List[nn.Module] = []
        gru_out_size = gru_units
        for units in fc_units:
            fc_layers += [nn.Linear(gru_out_size, units), nn.ReLU()]
            gru_out_size = units
        self.fc = nn.Sequential(*fc_layers)

        self.classifier = nn.Linear(gru_out_size, n_classes)
        self.last_act = _make_activation(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time=16, feature=96)
        x = x.permute(0, 2, 1)          # -> (batch, 96, 16)
        x = self.convs(x)               # -> (batch, Cn, T)
        x = x.permute(0, 2, 1)          # -> (batch, T, Cn) for GRU
        out, _ = self.gru(x)            # -> (batch, T, gru_units)
        x = out[:, -1, :]               # take last timestep
        x = self.dropout(x)
        x = self.fc(x)
        return self.last_act(self.classifier(x))


# ---------------------------------------------------------------------------
# 7. MobileNetV2Model  (port từ kws_streaming/mobilenet_v2.py)
#    Paper: https://arxiv.org/abs/1801.04381
# ---------------------------------------------------------------------------

class _InvertedResidual(nn.Module):
    """Inverted Residual Block của MobileNetV2 (1D version)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        expansion: float = 1.5,
    ) -> None:
        super().__init__()
        exp_ch = int(in_ch * expansion)
        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers: List[nn.Module] = [
            # Pointwise expansion
            nn.Conv1d(in_ch, exp_ch, 1, bias=False),
            nn.BatchNorm1d(exp_ch),
            nn.ReLU6(),
            # Depthwise
            nn.Conv1d(exp_ch, exp_ch, kernel_size,
                      stride=stride, padding=kernel_size // 2,
                      groups=exp_ch, bias=False),
            nn.BatchNorm1d(exp_ch),
            nn.ReLU6(),
            # Pointwise projection
            nn.Conv1d(exp_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            return out + x
        return out


class MobileNetV2Model(nn.Module):
    """MobileNetV2 (1D, reduced) for keyword spotting.

    Port từ kws_streaming/mobilenet_v2.py sang PyTorch.
    Paper: MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381

    Input : (batch, 16, 96)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        input_shape: tuple = (16, 96),
        n_classes: int = 1,
        stem_filters: int = 32,
        stem_kernel_size: int = 3,
        stem_stride: int = 2,
        block_filters: List[int] = None,
        block_kernel_sizes: List[int] = None,
        block_strides: List[int] = None,
        block_expansions: List[float] = None,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__()

        if block_filters is None:
            block_filters = [32, 32, 64, 64]
        if block_kernel_sizes is None:
            block_kernel_sizes = [3, 3, 3, 3]
        if block_strides is None:
            block_strides = [1, 2, 1, 1]
        if block_expansions is None:
            block_expansions = [1.5, 1.5, 1.5, 1.5]

        in_ch = input_shape[1]  # 96

        # Stem conv
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, stem_filters, stem_kernel_size,
                      stride=stem_stride, padding=stem_kernel_size // 2, bias=False),
            nn.BatchNorm1d(stem_filters),
            nn.ReLU6(),
        )
        in_ch = stem_filters

        blocks: List[nn.Module] = []
        for out_ch, ksize, stride, exp in zip(
            block_filters, block_kernel_sizes, block_strides, block_expansions
        ):
            blocks.append(_InvertedResidual(in_ch, out_ch, ksize, stride, exp))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_ch, n_classes)
        self.last_act = _make_activation(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)          # -> (batch, 96, 16)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.last_act(self.classifier(x))


# ---------------------------------------------------------------------------
# 8. InceptionModel  (port từ kws_streaming/inception.py)
#    Paper: http://arxiv.org/abs/1512.00567
# ---------------------------------------------------------------------------

class _InceptionBlock(nn.Module):
    """Inception block với 3 nhánh song song.

    branch1: Conv 1x1
    branch2: Conv 1x1 -> Conv kx1
    branch3: Conv 1x1 -> Conv kx1 -> Conv kx1
    Kết quả concat 3 nhánh -> Conv 1x1 để giảm chiều.
    """

    def __init__(self, in_ch: int, filters1: int,
                 filters2: int, kernel_size: int) -> None:
        super().__init__()
        pad = kernel_size // 2

        self.b1 = nn.Sequential(
            nn.Conv1d(in_ch, filters1, 1, bias=False),
            nn.BatchNorm1d(filters1), nn.ReLU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv1d(in_ch, filters1, 1, bias=False),
            nn.BatchNorm1d(filters1), nn.ReLU(),
            nn.Conv1d(filters1, filters1, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(filters1), nn.ReLU(),
        )
        self.b3 = nn.Sequential(
            nn.Conv1d(in_ch, filters1, 1, bias=False),
            nn.BatchNorm1d(filters1), nn.ReLU(),
            nn.Conv1d(filters1, filters1, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(filters1), nn.ReLU(),
            nn.Conv1d(filters1, filters1, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(filters1), nn.ReLU(),
        )
        # Bottleneck 1x1 để giảm từ 3*filters1 -> filters2
        self.bottleneck = nn.Sequential(
            nn.Conv1d(filters1 * 3, filters2, 1, bias=False),
            nn.BatchNorm1d(filters2), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.bottleneck(out)


class InceptionModel(nn.Module):
    """Inception model (reduced) for keyword spotting.

    Port từ kws_streaming/inception.py sang PyTorch.
    Paper: Rethinking the Inception Architecture for Computer Vision
    http://arxiv.org/abs/1512.00567

    Input : (batch, 16, 96)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        input_shape: tuple = (16, 96),
        n_classes: int = 1,
        stem_filters: List[int] = None,
        stem_kernels: List[int] = None,
        inception_filters1: List[int] = None,
        inception_filters2: List[int] = None,
        inception_kernels: List[int] = None,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__()

        if stem_filters is None:
            stem_filters = [24]
        if stem_kernels is None:
            stem_kernels = [5]
        if inception_filters1 is None:
            inception_filters1 = [10, 10, 16]
        if inception_filters2 is None:
            inception_filters2 = [10, 10, 16]
        if inception_kernels is None:
            inception_kernels = [5, 5, 5]

        in_ch = input_shape[1]  # 96

        stem_layers: List[nn.Module] = []
        for out_ch, ksize in zip(stem_filters, stem_kernels):
            pad = ksize // 2
            stem_layers += [
                nn.Conv1d(in_ch, out_ch, ksize, padding=pad, bias=False),
                nn.BatchNorm1d(out_ch), nn.ReLU(),
            ]
            in_ch = out_ch
        self.stem = nn.Sequential(*stem_layers)

        inc_blocks: List[nn.Module] = []
        for f1, f2, ksize in zip(inception_filters1, inception_filters2, inception_kernels):
            inc_blocks.append(_InceptionBlock(in_ch, f1, f2, ksize))
            in_ch = f2
        self.inception_blocks = nn.Sequential(*inc_blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_ch, n_classes)
        self.last_act = _make_activation(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)                    # -> (batch, 96, 16)
        x = self.stem(x)
        x = self.inception_blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.last_act(self.classifier(x))


# ---------------------------------------------------------------------------
# 9. InceptionResNetModel  (port từ kws_streaming/inception_resnet.py)
#    Paper: https://arxiv.org/abs/1602.07261
# ---------------------------------------------------------------------------

class _InceptionResBlock(nn.Module):
    """Inception-ResNet block: 2 nhánh song song + scaled residual."""

    def __init__(self, in_ch: int, branch0_ch: int,
                 branch1_ch: int, kernel_size: int, scale: float = 0.2) -> None:
        super().__init__()
        self.scale = scale
        pad = kernel_size // 2

        self.b0 = nn.Sequential(
            nn.Conv1d(in_ch, branch0_ch, 1, bias=False),
            nn.BatchNorm1d(branch0_ch), nn.ReLU(),
        )
        self.b1 = nn.Sequential(
            nn.Conv1d(in_ch, branch0_ch, 1, bias=False),
            nn.BatchNorm1d(branch0_ch), nn.ReLU(),
            nn.Conv1d(branch0_ch, branch1_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(branch1_ch), nn.ReLU(),
            nn.Conv1d(branch1_ch, branch1_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(branch1_ch), nn.ReLU(),
        )
        # 1x1 để đưa concat về cùng dim với input
        concat_ch = branch0_ch + branch1_ch
        self.project = nn.Sequential(
            nn.Conv1d(concat_ch, in_ch, 1, bias=True),
            nn.BatchNorm1d(in_ch),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = torch.cat([self.b0(x), self.b1(x)], dim=1)
        up = self.project(mixed)
        return self.act(x + up * self.scale)


class InceptionResNetModel(nn.Module):
    """Inception-ResNet model (reduced) for keyword spotting.

    Port từ kws_streaming/inception_resnet.py sang PyTorch.
    Paper: Inception-v4, Inception-ResNet and the Impact of Residual Connections
    https://arxiv.org/abs/1602.07261

    Input : (batch, 16, 96)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        input_shape: tuple = (16, 96),
        n_classes: int = 1,
        stem_filters: List[int] = None,
        stem_kernels: List[int] = None,
        scales: List[float] = None,
        branch0_filters: List[int] = None,
        branch1_filters: List[int] = None,
        out_filters: List[int] = None,
        block_kernels: List[int] = None,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__()

        if stem_filters is None:
            stem_filters = [32]
        if stem_kernels is None:
            stem_kernels = [5]
        if scales is None:
            scales = [0.2, 0.5, 1.0]
        if branch0_filters is None:
            branch0_filters = [32, 32, 32]
        if branch1_filters is None:
            branch1_filters = [32, 32, 32]
        if out_filters is None:
            out_filters = [32, 32, 64]
        if block_kernels is None:
            block_kernels = [3, 5, 5]

        in_ch = input_shape[1]

        stem_layers: List[nn.Module] = []
        for out_ch, ksize in zip(stem_filters, stem_kernels):
            pad = ksize // 2
            stem_layers += [
                nn.Conv1d(in_ch, out_ch, ksize, padding=pad, bias=False),
                nn.BatchNorm1d(out_ch), nn.ReLU(),
            ]
            in_ch = out_ch
        self.stem = nn.Sequential(*stem_layers)

        blocks: List[nn.Module] = []
        for scale, b0, b1, out_ch, ksize in zip(
            scales, branch0_filters, branch1_filters, out_filters, block_kernels
        ):
            blocks.append(_InceptionResBlock(in_ch, b0, b1, ksize, scale))
            # Bottleneck để đưa về out_ch
            blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch), nn.ReLU(),
            ))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_ch, n_classes)
        self.last_act = _make_activation(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.last_act(self.classifier(x))


# ---------------------------------------------------------------------------
# 10. SVDFResNetModel  (port từ kws_streaming/svdf_resnet.py)
#     Paper: https://arxiv.org/pdf/1812.02802.pdf
# ---------------------------------------------------------------------------

class _SVDFLayer(nn.Module):
    """Singular Value Decomposition Filter (SVDF) layer — 1D implementation.

    SVDF phân rã dense op thành 2 bước:
        1. DepthwiseConv1D theo trục thời gian (memory_size) - học thời gian
        2. Linear 1x1 theo trục feature                     - học đặc trưng
    Đây là dạng low-rank approximation của một lớp Dense thông thường.
    """

    def __init__(self, in_ch: int, out_ch: int, memory_size: int,
                 use_batch_norm: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        # Depthwise theo time (memory)
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size=memory_size,
            padding=memory_size - 1, groups=in_ch, bias=False,
        )
        self.memory_size = memory_size
        # Pointwise (feature mixing)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=True)
        self.bn = nn.BatchNorm1d(out_ch) if use_batch_norm else nn.Identity()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, ch, time)
        x = self.depthwise(x)[:, :, :x.shape[2]]  # causal: cắt bỏ phần thừa bên phải
        x = self.drop(self.bn(self.pointwise(x)))
        return x


class _SVDFResBlock(nn.Module):
    """Khối SVDF với residual connection."""

    def __init__(self, in_ch: int, units: List[int],
                 memory_sizes: List[int], pool: int,
                 use_batch_norm: bool = True,
                 dropout: float = 0.0, activation: str = "relu") -> None:
        super().__init__()
        act_map = {"relu": nn.ReLU, "gelu": nn.GELU}
        self.act = act_map.get(activation, nn.ReLU)()

        layers: List[nn.Module] = []
        ch = in_ch
        for i, (out_ch, mem) in enumerate(zip(units, memory_sizes)):
            layers.append(_SVDFLayer(ch, out_ch, mem, use_batch_norm, dropout))
            if i < len(units) - 1:  # activation giữa các lớp, lớp cuối dùng linear
                layers.append(act_map.get(activation, nn.ReLU)())
            ch = out_ch
        self.layers = nn.Sequential(*layers)

        # Shortcut: Dense 1x1 tương đương
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, units[-1], 1, bias=False),
            nn.BatchNorm1d(units[-1]),
        )
        self.pool = nn.MaxPool1d(pool, stride=pool) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        out = self.act(out + self.shortcut(x))
        return self.pool(out)


class SVDFResNetModel(nn.Module):
    """SVDF with Residual connections for keyword spotting.

    Port từ kws_streaming/svdf_resnet.py sang PyTorch.
    Paper: End-to-End Streaming Keyword Spotting
    https://arxiv.org/pdf/1812.02802.pdf

    SVDF: low-rank decomposition của Dense op — cực kỳ hiệu quả cho streaming
    trên các thiết bị nhúng vì chỉ cần cập nhật bộ nhớ từng bước thời gian.

    Input : (batch, 16, 96)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        input_shape: tuple = (16, 96),
        n_classes: int = 1,
        block_units: List[List[int]] = None,
        block_memory_sizes: List[List[int]] = None,
        blocks_pool: List[int] = None,
        use_batch_norm: bool = True,
        svdf_dropout: float = 0.0,
        dropout: float = 0.0,
        fc_units: List[int] = None,
        activation: str = "relu",
        **kwargs,
    ) -> None:
        super().__init__()

        if block_units is None:
            block_units = [[256, 256], [256, 256], [256, 256]]
        if block_memory_sizes is None:
            block_memory_sizes = [[4, 10], [10, 10], [10, 10]]
        if blocks_pool is None:
            blocks_pool = [1, 2, 2]
        if fc_units is None:
            fc_units = []

        in_ch = input_shape[1]  # 96
        svdf_blocks: List[nn.Module] = []
        for units, mems, pool in zip(block_units, block_memory_sizes, blocks_pool):
            svdf_blocks.append(
                _SVDFResBlock(in_ch, units, mems, pool, use_batch_norm, svdf_dropout, activation)
            )
            in_ch = units[-1]
        self.svdf_blocks = nn.Sequential(*svdf_blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)

        fc_layers: List[nn.Module] = []
        for units in fc_units:
            fc_layers += [nn.Linear(in_ch, units), nn.ReLU()]
            in_ch = units
        self.fc = nn.Sequential(*fc_layers)

        self.classifier = nn.Linear(in_ch, n_classes)
        self.last_act = _make_activation(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)          # -> (batch, 96, 16)
        x = self.svdf_blocks(x)          # -> (batch, units, T')
        x = self.pool(x).squeeze(-1)     # -> (batch, units)
        x = self.dropout(x)
        x = self.fc(x)
        return self.last_act(self.classifier(x))


# ---------------------------------------------------------------------------
# Registry: ánh xạ tên model -> class
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "dnn":               DNNModel,
    "rnn":               RNNModel,
    "cnn":               CNNModel,
    "crnn":              CRNNModel,
    "mobilenet_v2":      MobileNetV2Model,
    "inception":         InceptionModel,
    "inception_resnet":  InceptionResNetModel,
    "tc_resnet":         TCResNet,
    "ds_tc_resnet":      DSTCResNet,
    "svdf_resnet":       SVDFResNetModel,
}


def build_model(
    model_type: str,
    input_shape: tuple = (16, 96),
    n_classes: int = 1,
    **kwargs,
) -> nn.Module:
    """Factory function: tạo model từ tên.

    Args:
        model_type: Một trong các giá trị sau:
            ``"dnn"``, ``"rnn"``, ``"cnn"``, ``"crnn"``,
            ``"mobilenet_v2"``, ``"inception"``, ``"inception_resnet"``,
            ``"tc_resnet"``, ``"ds_tc_resnet"``, ``"svdf_resnet"``.
        input_shape: Shape đầu vào (frames, features). Mặc định (16, 96).
        n_classes: Số lớp đầu ra.
        **kwargs: Các tham số bổ sung truyền thẳng vào constructor của model.

    Returns:
        nn.Module đã được khởi tạo.

    Raises:
        ValueError: Nếu ``model_type`` không có trong registry.
    """
    model_type = model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"model_type='{model_type}' không được hỗ trợ. "
            f"Các lựa chọn hợp lệ: {list(MODEL_REGISTRY.keys())}"
        )
    cls = MODEL_REGISTRY[model_type]
    return cls(input_shape=input_shape, n_classes=n_classes, **kwargs)
