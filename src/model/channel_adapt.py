"""通道适配：FACED 32ch ckpt → TY 31ch 输入（按电极名）。

依据：runs/FACED_to_TY9_movie_zeroshot_cp/CHANNEL_ORDER_AUDIT.md
  - FACED 权重轴 = 预处理后标准名序（FACED_PROCESSED_NAMES）
  - TY 9_movie pkl 轴 = 真实 BDF 经 TY_CHANNEL_REORDER 后的实际名序
  - 按名精确/近似映射 → 重排 TY + 切片 msConv；FACED·CP2 无对应则丢弃

部署侧名表文件（channel_order*.csv）仅作历史/文档参考，运行时不按下标使用。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)

# TY 预处理里用的重排下标（与 Preprocessing_tongyong.TY_CHANNEL_REORDER 一致）
# 用于从真实 BDF 名序推断 pkl 轴上的电极名；不是「对齐 FACED」的依据
TY_CHANNEL_REORDER: List[int] = [
    0, 1, 2, 3, 4, 5, 6, 25, 10, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 27, 28, 23, 24, 26, 29, 30, 9,
]

# FACED 主线训练张量通道序（预处理后）
FACED_PROCESSED_NAMES: List[str] = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1",
    "FC2", "FC5", "FC6", "Cz", "C3", "C4", "T3", "T4",
    "CP1", "CP2", "CP5", "CP6", "Pz", "P3", "P4", "T5",
    "T6", "PO3", "PO4", "Oz", "O1", "O2", "A2", "A1",
]

# TY 真实 BDF 原始名序（多被试一致）
TY_BDF_RAW_NAMES: List[str] = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FCz",
    "FC3", "FC4", "FT7", "FT8", "Cz", "C3", "C4", "T3",
    "T4", "A1", "A2", "CP3", "CP4", "TP7", "TP8", "Pz",
    "P3", "P4", "T5", "T6", "Oz", "O1", "O2",
]

# 9_movie pkl 通道轴名序
TY_PKL_CHANNEL_NAMES: List[str] = [TY_BDF_RAW_NAMES[i] for i in TY_CHANNEL_REORDER]

TY_TO_FACED_NAME_APPROX: Dict[str, str] = {
    "FCz": "FC1",
    "FT8": "FC2",
    "FC3": "FC5",
    "FC4": "FC6",
    "CP3": "CP5",
    "CP4": "CP6",
    "TP7": "PO3",
    "TP8": "PO4",
    "FT7": "CP1",
}

CONV_LAYERS_TO_SLICE = [
    "model.msConv1.weight",
    "model.msConv2.weight",
    "model.msConv3.weight",
    "model.msConv4.weight",
]

DEFAULT_MODE = "ty_pkl_to_faced_by_name"


def _build_name_align_indices() -> Tuple[List[int], List[int], List[Tuple[int, str, int, str, str]]]:
    faced_index = {n: i for i, n in enumerate(FACED_PROCESSED_NAMES)}
    pairs: List[Tuple[int, int, str, str, str]] = []
    for ti, tn in enumerate(TY_PKL_CHANNEL_NAMES):
        target = TY_TO_FACED_NAME_APPROX.get(tn, tn)
        if target not in faced_index:
            raise KeyError(f"TY pkl channel {tn!r} -> {target!r} not in FACED_PROCESSED_NAMES")
        fi = faced_index[target]
        kind = "exact" if tn == target else f"approx({tn}->{target})"
        pairs.append((ti, fi, tn, FACED_PROCESSED_NAMES[fi], kind))
    pairs_sorted = sorted(pairs, key=lambda x: x[1])
    data_idx = [p[0] for p in pairs_sorted]
    weight_idx = [p[1] for p in pairs_sorted]
    detail = [(p[0], p[2], p[1], p[3], p[4]) for p in pairs_sorted]
    if len(set(weight_idx)) != len(weight_idx):
        raise ValueError("name align: duplicate FACED weight indices")
    return data_idx, weight_idx, detail


TY_PKL_TO_FACED_DATA_IDX, FACED_WEIGHT_SELECT_BY_NAME, NAME_ALIGN_DETAIL = _build_name_align_indices()


def reorder_data_by_idx(
    data: np.ndarray,
    data_idx: List[int],
    channel_axis: int = -2,
    expect_n: int = 31,
) -> np.ndarray:
    if data.shape[channel_axis] != expect_n:
        raise ValueError(
            f"channel reorder expects {expect_n} channels at axis {channel_axis}, "
            f"got shape {data.shape}"
        )
    idx = np.asarray(data_idx, dtype=np.int64)
    return np.take(data, idx, axis=channel_axis)


def reorder_data_ty_pkl_to_faced_by_name(data: np.ndarray, channel_axis: int = -2) -> np.ndarray:
    return reorder_data_by_idx(
        data, TY_PKL_TO_FACED_DATA_IDX, channel_axis=channel_axis, expect_n=31
    )


def slice_msconv_weights(
    state_dict: dict,
    weight_select_idx: List[int],
    layer_keys: Optional[List[str]] = None,
) -> Tuple[dict, List[Tuple[str, tuple, tuple]]]:
    if layer_keys is None:
        layer_keys = CONV_LAYERS_TO_SLICE
    idx_t = torch.tensor(weight_select_idx, dtype=torch.long)
    new_state = dict(state_dict)
    diffs = []
    for k in layer_keys:
        if k not in new_state:
            log.warning(f"channel_adapt: state_dict 缺少 {k}，跳过")
            continue
        w = new_state[k]
        if w.ndim != 4 or w.shape[2] < idx_t.max().item() + 1:
            log.warning(
                f"channel_adapt: 层 {k} 形状 {tuple(w.shape)} 不匹配，跳过切片"
            )
            continue
        w_new = w.index_select(dim=2, index=idx_t).contiguous()
        diffs.append((k, tuple(w.shape), tuple(w_new.shape)))
        new_state[k] = w_new
    return new_state, diffs


def adapt_extractor_inplace(
    extractor: torch.nn.Module,
    weight_select_idx: List[int],
    target_n_channs: int,
) -> List[Tuple[str, tuple, tuple]]:
    import torch.nn as nn

    inner = extractor.model
    diffs = []
    idx_t = torch.tensor(weight_select_idx, dtype=torch.long)
    for layer_name in ["msConv1", "msConv2", "msConv3", "msConv4"]:
        if not hasattr(inner, layer_name):
            log.warning(f"channel_adapt: model 没有 {layer_name}，跳过")
            continue
        old: nn.Conv2d = getattr(inner, layer_name)
        old_w = old.weight.data
        if old_w.shape[2] < idx_t.max().item() + 1:
            log.warning(
                f"channel_adapt: {layer_name} 通道维 {old_w.shape[2]} "
                f"< {idx_t.max().item()+1}"
            )
            continue
        new_layer = nn.Conv2d(
            in_channels=old.in_channels,
            out_channels=old.out_channels,
            kernel_size=(target_n_channs, old.kernel_size[1]),
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=(old.bias is not None),
        )
        new_layer.weight.data = old_w.index_select(dim=2, index=idx_t).contiguous().clone()
        if old.bias is not None:
            new_layer.bias.data = old.bias.data.clone()
        new_layer.to(old.weight.device, dtype=old.weight.dtype)
        setattr(inner, layer_name, new_layer)
        diffs.append((layer_name, tuple(old_w.shape), tuple(new_layer.weight.shape)))

    log.info(
        f"channel_adapt: target_n_channs={target_n_channs}, "
        f"selected {len(weight_select_idx)} idx"
    )
    for name, oshape, nshape in diffs:
        log.info(f"  {name}: {oshape} -> {nshape}")
    return diffs


def get_preset_select_idx(mode: str = DEFAULT_MODE) -> Tuple[List[int], List[int], int]:
    """返回 (data_idx, weight_idx, target_n_channs)。"""
    if mode in (DEFAULT_MODE, "ty_pkl_to_faced_by_name"):
        return TY_PKL_TO_FACED_DATA_IDX, FACED_WEIGHT_SELECT_BY_NAME, 31
    raise ValueError(
        f"channel_adapt: 未知 mode={mode}（仅支持 {DEFAULT_MODE}）"
    )


def apply_warm_start_from_faced_ckpt(
    extractor: torch.nn.Module,
    faced_ckpt_path: str,
    mode: str = DEFAULT_MODE,
    map_location: str = "cpu",
) -> dict:
    _, weight_idx_list, target_n_channs = get_preset_select_idx(mode)
    raw = torch.load(faced_ckpt_path, map_location=map_location, weights_only=False)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    sliced_state, diffs = slice_msconv_weights(state, weight_idx_list)
    missing, unexpected = extractor.load_state_dict(sliced_state, strict=False)
    target_keys = set(extractor.state_dict().keys())
    matched_n = len(target_keys) - len(missing)
    info = {
        "ckpt_path": str(faced_ckpt_path),
        "mode": mode,
        "target_n_channs": target_n_channs,
        "sliced_layers": diffs,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "matched_keys_n": int(matched_n),
        "total_target_keys_n": int(len(target_keys)),
    }
    log.info(
        f"warm_start: ckpt={faced_ckpt_path} matched={matched_n}/{len(target_keys)} "
        f"missing={len(missing)} unexpected={len(unexpected)} sliced={len(diffs)}"
    )
    return info
