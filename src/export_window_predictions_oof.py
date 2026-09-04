#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export out-of-fold (OOF) per-window predictions for DAEST-ME.

Anti-leakage rules (must hold):
1. Fold k uses ONLY fold-k feature file (*_f{k}_fea_{mode}.npy) and fold-k MLP.
2. Fold k predicts ONLY that fold's validation subjects (same split as train/visualize).
3. Each subject index appears in exactly one validation fold (full OOF cover).
4. Does NOT retrain; does NOT score train subjects with the same-fold model.

Downstream window meta (DAEST mainline): timeLen2=1s, timeStep2=1s, trial=30s
→ typically 30 windows / video.

Presets:
  faced              → FACED 10-fold, matched fea_r3 + mlp r3 (~59.9%)
  ty9_movie           → TY9_movie 5-fold, fea_r1 + mlp r1
  ty9_communication  → TY9_communication 5-fold, fea_r1 + mlp r1
  ty9_movie_matched51 → TY9_movie_matched51 5-fold（双任务共同 51 人正式重训）
  ty9_communication_matched51 → TY9_communication_matched51 5-fold
  ty8_movie_self          → TY8_movie_self 5-fold；逐被试自评标签；y_true=-1 不计分
  ty8_communication_self  → TY8_communication_self 5-fold；自评；y_true=-1 不计分
  faced_to_ty9_movie_mlp_ft → E08 FACED→TY观影 MLP-FT（冻结 encoder）
  faced_to_ty9_communication_mlp_ft → D10 FACED→TY交流 MLP-FT
  ty9_movie_to_ty9_communication_mlp_ft_matched51 → E10 观影→交流 MLP-FT matched51

Usage (from FACED-base):
  python export_window_predictions_oof.py --preset faced
  python export_window_predictions_oof.py --preset ty9_movie
  CUDA_VISIBLE_DEVICES=5 python export_window_predictions_oof.py --preset ty9_communication
  CUDA_VISIBLE_DEVICES=1 python export_window_predictions_oof.py --preset ty9_movie_matched51
  CUDA_VISIBLE_DEVICES=2 python export_window_predictions_oof.py --preset ty9_communication_matched51
  CUDA_VISIBLE_DEVICES=5 python export_window_predictions_oof.py --preset ty8_movie_self
  CUDA_VISIBLE_DEVICES=0 python export_window_predictions_oof.py --preset ty8_communication_self
  python export_window_predictions_oof.py --preset faced_to_ty9_movie_mlp_ft --device cpu
  python export_window_predictions_oof.py --preset faced_to_ty9_communication_mlp_ft --device cpu
  python export_window_predictions_oof.py --preset ty9_movie_to_ty9_communication_mlp_ft_matched51 --device cpu
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import PDataset
from model.pl_models import MLPModel

CLASS_NAMES_9 = [
    "anger",
    "disgust",
    "fear",
    "sadness",
    "neutral",
    "amusement",
    "inspiration",
    "joy",
    "tenderness",
]
CLASS_NAMES_8 = [
    "anger",
    "disgust",
    "fear",
    "sadness",
    "amusement",
    "inspiration",
    "joy",
    "tenderness",
]


@dataclass(frozen=True)
class ExportConfig:
    name: str
    n_subs: int
    n_folds: int
    n_class: int
    n_vids: int
    save_name: str
    feat_dir: str
    cp_dir: str
    run: int
    mode: str
    out_dir: str
    subject_ids_pkl: Optional[str]
    time_len2: float = 1.0
    time_step2: float = 1.0
    trial_sec: float = 30.0
    ckpt_pick: str = "mtime"  # mtime | first
    # True：用 allsubs_label2.npy（逐被试标签，含自评 -1）
    use_allsubs_labels: bool = False
    # True：y_true==-1 不进 overall/被试/窗准确率（仍写入 CSV）
    ignore_invalid: bool = False
    class_names: Optional[tuple] = None


PRESETS = {
    "faced": ExportConfig(
        name="FACED_05_47",
        n_subs=123,
        n_folds=10,
        n_class=9,
        n_vids=28,
        save_name="FACED",
        # fea_r1 已缺失；fea_r3 + mlp r3 与现成 ~59.9% 图一致
        feat_dir="/data/liujialing/TY/建模/被试间/DAEST/黎曼/FACED-base/runs/FACED_05_47_cp/fea_r3",
        cp_dir="/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/runs/FACED_05_47_cp/runs/cp",
        run=3,
        mode="me",
        out_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/FACED_05_47_cp/runs/window_predictions_oof_r3"
        ),
        subject_ids_pkl=None,
        ckpt_pick="first",  # 与 visualize_faced_05_47.py 一致
    ),
    "ty9_movie": ExportConfig(
        name="TY9_movie",
        n_subs=52,
        n_folds=5,
        n_class=9,
        n_vids=28,
        save_name="TY9_movie",
        feat_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_movie_ME_cp/runs/TY9_movie/ext_fea/fea_r1"
        ),
        cp_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_movie_ME_cp/runs/cp"
        ),
        run=1,
        mode="me",
        out_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_movie_ME_cp/runs/window_predictions_oof_r1"
        ),
        subject_ids_pkl=(
            "/data/liujialing/TY/预处理/python/Preprocessing/全部的TY/output/9_movie/"
            "data_9class_movie.pkl"
        ),
        ckpt_pick="mtime",
    ),
    "ty9_communication": ExportConfig(
        name="TY9_communication",
        n_subs=118,
        n_folds=5,
        n_class=9,
        n_vids=28,
        save_name="TY9_communication",
        feat_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_communication_ME_cp/runs/TY9_communication/ext_fea/fea_r1"
        ),
        cp_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_communication_ME_cp/runs/cp"
        ),
        run=1,
        mode="me",
        out_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_communication_ME_cp/runs/window_predictions_oof_r1"
        ),
        subject_ids_pkl=(
            "/data/liujialing/TY/预处理/python/Preprocessing/全部的TY/output/9_communication/"
            "data_9class_communication.pkl"
        ),
        ckpt_pick="mtime",
    ),
    # 2026-08-31 新增（纯增量，未改动任何既有条目）
    # TY 交流 · 素材标签 · 去掉中性 → 8 类。复用 A 版 9 类特征（软链只读）。
    # onesub 素材标签（use_allsubs_labels=False）+ 中性 -1 不进指标（ignore_invalid=True）
    "ty9_communication_8class": ExportConfig(
        name="TY9_communication_8class",
        n_subs=118,
        n_folds=5,
        n_class=8,
        n_vids=28,
        save_name="TY9_communication_8class",
        feat_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_communication_8class_A_cp/runs/TY9_communication_8class/ext_fea/fea_r1"
        ),
        cp_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_communication_8class_A_cp/runs/cp"
        ),
        run=1,
        mode="me",
        out_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_communication_8class_A_cp/runs/window_predictions_oof_r1"
        ),
        subject_ids_pkl=(
            "/data/liujialing/TY/预处理/python/Preprocessing/全部的TY/output/9_communication/"
            "data_9class_communication.pkl"
        ),
        ckpt_pick="mtime",
        use_allsubs_labels=False,
        ignore_invalid=True,
        class_names=tuple(CLASS_NAMES_8),
    ),
    "ty8_movie_self": ExportConfig(
        name="TY8_movie_self",
        n_subs=52,
        n_folds=5,
        n_class=8,
        n_vids=28,
        save_name="TY8_movie_self",
        feat_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY8_movie_self_ME_cp/runs/TY8_movie_self/ext_fea/fea_r1"
        ),
        cp_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY8_movie_self_ME_cp/runs/cp"
        ),
        run=1,
        mode="me",
        out_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY8_movie_self_ME_cp/runs/window_predictions_oof_r1"
        ),
        subject_ids_pkl=(
            "/data/liujialing/TY/预处理/python/Preprocessing/全部的TY/output/8_movie_self/"
            "data_8class_movie_self.pkl"
        ),
        ckpt_pick="mtime",
        use_allsubs_labels=True,
        ignore_invalid=True,
        class_names=tuple(CLASS_NAMES_8),
    ),
    "ty8_communication_self": ExportConfig(
        name="TY8_communication_self",
        n_subs=118,
        n_folds=5,
        n_class=8,
        n_vids=28,
        save_name="TY8_communication_self",
        feat_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY8_communication_self_ME_cp/runs/TY8_communication_self/ext_fea/fea_r1"
        ),
        cp_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY8_communication_self_ME_cp/runs/cp"
        ),
        run=1,
        mode="me",
        out_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY8_communication_self_ME_cp/runs/window_predictions_oof_r1"
        ),
        subject_ids_pkl=(
            "/data/liujialing/TY/预处理/python/Preprocessing/全部的TY/output/8_communication_self/"
            "data_8class_communication_self.pkl"
        ),
        ckpt_pick="mtime",
        use_allsubs_labels=True,
        ignore_invalid=True,
        class_names=tuple(CLASS_NAMES_8),
    ),
    "ty9_movie_matched51": ExportConfig(
        name="TY9_movie_matched51",
        n_subs=51,
        n_folds=5,
        n_class=9,
        n_vids=28,
        save_name="TY9_movie_matched51",
        feat_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_movie_matched51_ME_cp/runs/TY9_movie_matched51/ext_fea/fea_r1"
        ),
        cp_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_movie_matched51_ME_cp/runs/cp"
        ),
        run=1,
        mode="me",
        out_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_movie_matched51_ME_cp/runs/window_predictions_oof_r1"
        ),
        subject_ids_pkl=(
            "/data/liujialing/TY/预处理/python/Preprocessing/全部的TY/output/9_movie_matched51/"
            "data_9class_movie_matched51.pkl"
        ),
        ckpt_pick="mtime",
    ),
    "ty9_communication_matched51": ExportConfig(
        name="TY9_communication_matched51",
        n_subs=51,
        n_folds=5,
        n_class=9,
        n_vids=28,
        save_name="TY9_communication_matched51",
        feat_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_communication_matched51_ME_cp/runs/TY9_communication_matched51/ext_fea/fea_r1"
        ),
        cp_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_communication_matched51_ME_cp/runs/cp"
        ),
        run=1,
        mode="me",
        out_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_communication_matched51_ME_cp/runs/window_predictions_oof_r1"
        ),
        subject_ids_pkl=(
            "/data/liujialing/TY/预处理/python/Preprocessing/全部的TY/output/9_communication_matched51/"
            "data_9class_communication_matched51.pkl"
        ),
        ckpt_pick="mtime",
    ),
    # --- transfer MLP-FT（冻结源域 encoder；目标域 5 折 OOF 逐窗）---
    "faced_to_ty9_movie_mlp_ft": ExportConfig(
        name="FACED_to_TY9_movie_mlp_ft",
        n_subs=52,
        n_folds=5,
        n_class=9,
        n_vids=28,
        save_name="TY9_movie",
        feat_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/FACED_to_TY9_movie_mlp_ft_cp/runs/TY9_movie/ext_fea/fea_r1"
        ),
        cp_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/FACED_to_TY9_movie_mlp_ft_cp/runs/cp"
        ),
        run=1,
        mode="me",
        out_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/FACED_to_TY9_movie_mlp_ft_cp/runs/window_predictions_oof_r1"
        ),
        subject_ids_pkl=(
            "/data/liujialing/TY/预处理/python/Preprocessing/全部的TY/output/9_movie/"
            "data_9class_movie.pkl"
        ),
        ckpt_pick="mtime",
    ),
    "faced_to_ty9_communication_mlp_ft": ExportConfig(
        name="FACED_to_TY9_communication_mlp_ft",
        n_subs=118,
        n_folds=5,
        n_class=9,
        n_vids=28,
        save_name="TY9_communication",
        feat_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/FACED_to_TY9_communication_mlp_ft_cp/runs/TY9_communication/ext_fea/fea_r1"
        ),
        cp_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/FACED_to_TY9_communication_mlp_ft_cp/runs/cp"
        ),
        run=1,
        mode="me",
        out_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/FACED_to_TY9_communication_mlp_ft_cp/runs/window_predictions_oof_r1"
        ),
        subject_ids_pkl=(
            "/data/liujialing/TY/预处理/python/Preprocessing/全部的TY/output/9_communication/"
            "data_9class_communication.pkl"
        ),
        ckpt_pick="mtime",
    ),
    "ty9_movie_to_ty9_communication_mlp_ft_matched51": ExportConfig(
        name="TY9_movie_to_TY9_communication_mlp_ft_matched51",
        n_subs=51,
        n_folds=5,
        n_class=9,
        n_vids=28,
        save_name="TY9_communication_matched51",
        feat_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_movie_to_TY9_communication_mlp_ft_matched51_cp/"
            "runs/TY9_communication_matched51/ext_fea/fea_r1"
        ),
        cp_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_movie_to_TY9_communication_mlp_ft_matched51_cp/runs/cp"
        ),
        run=1,
        mode="me",
        out_dir=(
            "/data/liujialing/TY/建模/被试间/DAEST/脑电/all/FACED-base/"
            "runs/TY9_movie_to_TY9_communication_mlp_ft_matched51_cp/"
            "runs/window_predictions_oof_r1"
        ),
        subject_ids_pkl=(
            "/data/liujialing/TY/预处理/python/Preprocessing/全部的TY/output/9_communication_matched51/"
            "data_9class_communication_matched51.pkl"
        ),
        ckpt_pick="mtime",
    ),
}


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def fold_val_subs(n_subs: int, n_folds: int, fold: int) -> np.ndarray:
    n_per = round(n_subs / n_folds)
    if fold < n_folds - 1:
        return np.arange(n_per * fold, n_per * (fold + 1))
    return np.arange(n_per * fold, n_subs)


def assert_oof_partition(n_subs: int, n_folds: int) -> dict[int, int]:
    """Return subject_index -> fold_id; raise if overlap/missing."""
    owner: dict[int, int] = {}
    for fold in range(n_folds):
        for s in fold_val_subs(n_subs, n_folds, fold):
            s = int(s)
            if s in owner:
                raise RuntimeError(f"OOF leak: subject {s} in fold {owner[s]} and {fold}")
            owner[s] = fold
    missing = [i for i in range(n_subs) if i not in owner]
    if missing:
        raise RuntimeError(f"OOF incomplete: missing subjects {missing[:10]}...")
    return owner


def load_subject_ids(cfg: ExportConfig) -> list[str]:
    if cfg.subject_ids_pkl and os.path.isfile(cfg.subject_ids_pkl):
        with open(cfg.subject_ids_pkl, "rb") as f:
            obj = pickle.load(f)
        ids = [str(x) for x in obj["subject_ids"]]
        if len(ids) < cfg.n_subs:
            raise ValueError(f"pkl subject_ids={len(ids)} < n_subs={cfg.n_subs}")
        if len(ids) > cfg.n_subs:
            ids = ids[: cfg.n_subs]
        return ids
    return [f"{i:03d}" for i in range(cfg.n_subs)]


def pick_mlp_ckpt(mlp_dir: str, fold: int, how: str) -> str:
    pats = [
        os.path.join(mlp_dir, f"mlp_f{fold}_wd=*.ckpt"),
        os.path.join(mlp_dir, f"mlp_f{fold}_*.ckpt"),
    ]
    ckpts: list[str] = []
    for pat in pats:
        ckpts = [p for p in glob.glob(pat) if "-v1" not in os.path.basename(p)]
        if ckpts:
            break
    if not ckpts:
        raise FileNotFoundError(f"No MLP checkpoint for fold {fold} under {mlp_dir}")
    if how == "first":
        return sorted(ckpts)[0]
    return max(ckpts, key=os.path.getmtime)


def infer_windows_per_vid(onesub_label2: np.ndarray, n_vids: int) -> int:
    n = len(onesub_label2)
    if n % n_vids != 0:
        raise ValueError(f"onesub_label2 length {n} not divisible by n_vids={n_vids}")
    return n // n_vids


def class_name_of(y: int, class_names: list[str]) -> str:
    if y < 0:
        return "INVALID"
    return class_names[y]


def export_one(cfg: ExportConfig, device: str, write_probs: bool = True) -> dict:
    owner = assert_oof_partition(cfg.n_subs, cfg.n_folds)
    subject_ids = load_subject_ids(cfg)
    if cfg.class_names is not None:
        class_names = list(cfg.class_names)
    elif cfg.n_class == 8:
        class_names = list(CLASS_NAMES_8)
    else:
        class_names = CLASS_NAMES_9[: cfg.n_class]
    if len(class_names) != cfg.n_class:
        raise ValueError(f"class_names={len(class_names)} != n_class={cfg.n_class}")

    lab_path = os.path.join(cfg.feat_dir, "onesub_label2.npy")
    onesub_label2 = np.load(lab_path)
    n_per_sub = len(onesub_label2)
    win_per_vid = infer_windows_per_vid(onesub_label2, cfg.n_vids)
    expected_win = int((cfg.trial_sec - cfg.time_len2) / cfg.time_step2) + 1
    if win_per_vid != expected_win:
        print(
            f"WARNING: windows_per_vid={win_per_vid}, "
            f"expected {expected_win} from timeLen2={cfg.time_len2}/timeStep2={cfg.time_step2}"
        )

    allsubs_labels = None
    if cfg.use_allsubs_labels:
        all_path = os.path.join(cfg.feat_dir, "allsubs_label2.npy")
        if not os.path.isfile(all_path):
            raise FileNotFoundError(f"need allsubs_label2.npy for {cfg.name}: {all_path}")
        allsubs_labels = np.load(all_path)
        if allsubs_labels.shape != (cfg.n_subs, n_per_sub):
            raise ValueError(
                f"allsubs_label2 shape={allsubs_labels.shape}, "
                f"expected ({cfg.n_subs}, {n_per_sub})"
            )
        print(
            f"[{cfg.name}] using allsubs_label2; "
            f"invalid(-1)={(allsubs_labels < 0).sum()} / {allsubs_labels.size}"
        )

    mlp_dir = os.path.join(cfg.cp_dir, cfg.save_name, f"r{cfg.run}")
    os.makedirs(cfg.out_dir, exist_ok=True)

    rows: list[dict] = []
    subject_correct = np.zeros(cfg.n_subs, dtype=np.int64)
    subject_total = np.zeros(cfg.n_subs, dtype=np.int64)
    fold_stats = []

    for fold in range(cfg.n_folds):
        val_subs = fold_val_subs(cfg.n_subs, cfg.n_folds, fold)
        # hard guard: only val subjects
        for s in val_subs:
            if owner[int(s)] != fold:
                raise RuntimeError("internal OOF owner mismatch")

        fea_pat = os.path.join(cfg.feat_dir, f"*_f{fold}_fea_{cfg.mode}.npy")
        files = sorted(glob.glob(fea_pat))
        if not files:
            raise FileNotFoundError(f"No feature file for fold {fold}: {fea_pat}")
        fea_path = files[0]
        # filename must contain this fold id
        if f"_f{fold}_fea_" not in os.path.basename(fea_path):
            raise RuntimeError(f"Feature file fold mismatch: {fea_path}")

        data2 = np.load(fea_path)
        if np.isnan(data2).any():
            data2 = np.nan_to_num(data2, nan=0.0)
        if data2.shape[0] % cfg.n_subs != 0:
            raise ValueError(
                f"Feature rows {data2.shape[0]} not divisible by n_subs={cfg.n_subs}"
            )
        fea_dim = data2.shape[-1]
        data2 = data2.reshape(cfg.n_subs, -1, fea_dim)
        if data2.shape[1] != n_per_sub:
            raise ValueError(
                f"fold {fold}: samples/sub={data2.shape[1]} != label len {n_per_sub}"
            )

        # ONLY validation subjects
        val_data = data2[val_subs].reshape(-1, fea_dim)
        if allsubs_labels is not None:
            labels_val = allsubs_labels[val_subs].reshape(-1).astype(np.int64)
        else:
            labels_val = np.tile(onesub_label2, len(val_subs)).astype(np.int64)

        ckpt_path = pick_mlp_ckpt(mlp_dir, fold, cfg.ckpt_pick)
        predictor = MLPModel.load_from_checkpoint(ckpt_path, map_location=device)
        predictor.eval()
        with torch.no_grad():
            probe = torch.zeros(1, fea_dim, device=device)
            out_dim = int(predictor(probe).shape[-1])
        if out_dim != cfg.n_class:
            raise ValueError(f"MLP out_dim={out_dim}, expected {cfg.n_class}: {ckpt_path}")

        loader = DataLoader(
            PDataset(val_data, labels_val),
            batch_size=512,
            shuffle=False,
            num_workers=0,
        )
        preds, trues, probs = [], [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits = predictor(x)
                pred = logits.argmax(1).cpu().numpy()
                preds.append(pred)
                trues.append(y.numpy())
                if write_probs:
                    probs.append(F.softmax(logits, dim=-1).cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        prob_mat = np.concatenate(probs) if write_probs else None

        if len(preds) != len(val_subs) * n_per_sub:
            raise RuntimeError("prediction count mismatch")

        if cfg.ignore_invalid:
            valid_mask = trues >= 0
            n_eval = int(valid_mask.sum())
            correct_fold = int(((preds == trues) & valid_mask).sum())
            fold_acc = float(correct_fold / n_eval) if n_eval else float("nan")
        else:
            n_eval = int(len(preds))
            correct_fold = int((preds == trues).sum())
            fold_acc = float(correct_fold / n_eval)

        fold_stats.append(
            {
                "fold": fold,
                "n_val_subs": int(len(val_subs)),
                "val_subs": [int(x) for x in val_subs.tolist()],
                "n_windows": int(len(preds)),
                "n_eval_windows": n_eval,
                "acc": fold_acc,
                "feature_file": os.path.basename(fea_path),
                "mlp_ckpt": os.path.basename(ckpt_path),
            }
        )

        for i in range(len(preds)):
            sub_local = i // n_per_sub
            within = i % n_per_sub
            sub_idx = int(val_subs[sub_local])
            # never export if this subject is not owned by this fold
            if owner[sub_idx] != fold:
                raise RuntimeError("attempted to export non-val subject")

            vid_pos = within // win_per_vid  # 0-based
            win_in_vid = within % win_per_vid
            y_true = int(trues[i])
            y_pred = int(preds[i])
            t0 = win_in_vid * cfg.time_step2
            t1 = t0 + cfg.time_len2
            valid = (not cfg.ignore_invalid) or (y_true >= 0)
            row = {
                "dataset": cfg.name,
                "run": cfg.run,
                "fold": fold,
                "subject_index": sub_idx,
                "subject_id": subject_ids[sub_idx],
                "video_index": vid_pos + 1,
                "video_pos0": vid_pos,
                "window_in_video": win_in_vid,
                "window_global_in_subject": within,
                "time_start_sec": t0,
                "time_end_sec": t1,
                "y_true": y_true,
                "y_true_name": class_name_of(y_true, class_names),
                "y_pred": y_pred,
                "y_pred_name": class_name_of(y_pred, class_names),
                "correct": int(y_true == y_pred) if valid else -1,
                "valid_for_metric": int(valid),
                "eval_split": "val_oof",
                "feature_mode": cfg.mode,
                "time_len2_sec": cfg.time_len2,
                "time_step2_sec": cfg.time_step2,
            }
            if prob_mat is not None:
                row["prob_max"] = float(prob_mat[i, y_pred])
                for c in range(cfg.n_class):
                    row[f"prob_{c}"] = float(prob_mat[i, c])
            rows.append(row)
            if valid:
                subject_total[sub_idx] += 1
                if y_true == y_pred:
                    subject_correct[sub_idx] += 1

        print(
            f"[{cfg.name}] fold {fold}: val_subs={len(val_subs)} "
            f"acc={100.0 * fold_acc:.2f}% (n_eval={n_eval}/{len(preds)}) "
            f"fea={os.path.basename(fea_path)} ckpt={os.path.basename(ckpt_path)}"
        )

    # each subject must have predictions counted; if ignore_invalid, allow all-invalid subjects
    if np.any(subject_total == 0):
        bad = np.where(subject_total == 0)[0].tolist()
        if cfg.ignore_invalid:
            print(
                f"WARNING: subjects with zero valid windows (all -1), "
                f"excluded from subject-mean: {bad}"
            )
        else:
            raise RuntimeError(f"subjects without OOF preds: {bad}")

    metric_rows = [r for r in rows if r.get("valid_for_metric", 1) == 1]
    overall_acc = float(np.mean([r["correct"] for r in metric_rows])) if metric_rows else float("nan")
    valid_sub_mask = subject_total > 0
    sub_acc = np.full(cfg.n_subs, np.nan, dtype=np.float64)
    sub_acc[valid_sub_mask] = 100.0 * subject_correct[valid_sub_mask] / subject_total[valid_sub_mask]
    mean_sub = float(np.nanmean(sub_acc) / 100.0) if np.any(valid_sub_mask) else float("nan")
    std_sub = float(np.nanstd(sub_acc) / 100.0) if np.any(valid_sub_mask) else float("nan")
    summary = {
        "dataset": cfg.name,
        "run": cfg.run,
        "n_subs": cfg.n_subs,
        "n_folds": cfg.n_folds,
        "n_class": cfg.n_class,
        "n_vids": cfg.n_vids,
        "windows_per_video": win_per_vid,
        "time_len2_sec": cfg.time_len2,
        "time_step2_sec": cfg.time_step2,
        "n_rows": len(rows),
        "n_eval_rows": len(metric_rows),
        "n_subjects_with_valid": int(valid_sub_mask.sum()),
        "subjects_all_invalid": [int(i) for i in np.where(~valid_sub_mask)[0].tolist()],
        "ignore_invalid": cfg.ignore_invalid,
        "use_allsubs_labels": cfg.use_allsubs_labels,
        "overall_acc": overall_acc,
        "mean_subject_acc": mean_sub,
        "std_subject_acc": std_sub,
        "chance": 1.0 / cfg.n_class,
        "feat_dir": cfg.feat_dir,
        "cp_dir": cfg.cp_dir,
        "anti_leakage": {
            "protocol": "subject-wise OOF: fold-k model + fold-k features → fold-k val subjects only",
            "each_subject_in_exactly_one_val_fold": True,
            "train_subjects_not_scored_by_same_fold_model": True,
        },
        "folds": fold_stats,
    }

    # write CSV
    csv_path = os.path.join(cfg.out_dir, f"{cfg.name.lower()}_window_predictions_oof_r{cfg.run}.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # subject summary
    sub_csv = os.path.join(cfg.out_dir, f"{cfg.name.lower()}_subject_accuracy_oof_r{cfg.run}.csv")
    with open(sub_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject_index", "subject_id", "fold", "n_windows", "n_correct", "accuracy"])
        for i in range(cfg.n_subs):
            acc_str = "" if not np.isfinite(sub_acc[i]) else f"{sub_acc[i] / 100.0:.6f}"
            w.writerow(
                [
                    i,
                    subject_ids[i],
                    owner[i],
                    int(subject_total[i]),
                    int(subject_correct[i]),
                    acc_str,
                ]
            )

    # window-index curve (aggregate over subjects/videos); skip invalid
    by_win = np.zeros(win_per_vid)
    by_win_n = np.zeros(win_per_vid)
    by_cls_win = np.zeros((cfg.n_class, win_per_vid))
    by_cls_win_n = np.zeros((cfg.n_class, win_per_vid))
    for r in metric_rows:
        wi = int(r["window_in_video"])
        by_win[wi] += r["correct"]
        by_win_n[wi] += 1
        yt = int(r["y_true"])
        if 0 <= yt < cfg.n_class:
            by_cls_win[yt, wi] += r["correct"]
            by_cls_win_n[yt, wi] += 1
    win_acc = np.divide(by_win, np.maximum(by_win_n, 1))
    cls_win_acc = np.divide(by_cls_win, np.maximum(by_cls_win_n, 1))

    win_csv = os.path.join(cfg.out_dir, f"{cfg.name.lower()}_acc_by_window_r{cfg.run}.csv")
    with open(win_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window_in_video", "time_start_sec", "time_end_sec", "n", "accuracy"])
        for wi in range(win_per_vid):
            w.writerow(
                [
                    wi,
                    wi * cfg.time_step2,
                    wi * cfg.time_step2 + cfg.time_len2,
                    int(by_win_n[wi]),
                    f"{win_acc[wi]:.6f}",
                ]
            )

    # plots
    fig, ax = plt.subplots(figsize=(10, 4))
    t_centers = np.arange(win_per_vid) * cfg.time_step2 + cfg.time_len2 / 2.0
    ax.plot(t_centers, 100.0 * win_acc, marker="o", linewidth=1.5)
    ax.axhline(100.0 / cfg.n_class, color="red", linestyle="--", label="chance")
    ax.set_xlabel("Time in trial (s, window center)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"{cfg.name} OOF accuracy by 1s window")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(cfg.out_dir, f"{cfg.name.lower()}_acc_by_window_r{cfg.run}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(100.0 * cls_win_acc, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_yticks(range(cfg.n_class))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("window_in_video (1s step)")
    ax.set_ylabel("True class")
    ax.set_title(f"{cfg.name} OOF class × window accuracy (%)")
    plt.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    heat_path = os.path.join(cfg.out_dir, f"{cfg.name.lower()}_cls_window_heatmap_r{cfg.run}.png")
    fig.savefig(heat_path, dpi=150)
    plt.close(fig)

    summary_path = os.path.join(cfg.out_dir, f"{cfg.name.lower()}_oof_summary_r{cfg.run}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    readme = os.path.join(cfg.out_dir, "README.md")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            f"# {cfg.name} OOF 逐窗预测导出\n\n"
            f"- 协议：跨被试 {cfg.n_folds} 折 OOF；每折只用该折特征文件 + 该折 MLP，预测该折验证被试。\n"
            f"- 窗长：下游 `{cfg.time_len2}s` / 步长 `{cfg.time_step2}s`（每视频 {win_per_vid} 窗）。\n"
            f"- use_allsubs_labels={cfg.use_allsubs_labels}；ignore_invalid={cfg.ignore_invalid}\n"
            f"- overall_acc = {100 * overall_acc:.2f}%（n_eval={len(metric_rows)}/{len(rows)}）；"
            f"mean_subject_acc = {np.nanmean(sub_acc):.2f}% ± {np.nanstd(sub_acc):.2f}% "
            f"（有效被试 {int(valid_sub_mask.sum())}/{cfg.n_subs}）\n"
            f"- 主表：`{os.path.basename(csv_path)}`\n"
            f"- 特征：`{cfg.feat_dir}`\n"
            f"- ckpt：`{mlp_dir}`\n"
        )

    print(
        f"\n[{cfg.name}] DONE overall={100 * overall_acc:.2f}% "
        f"mean_subject={np.nanmean(sub_acc):.2f}%±{np.nanstd(sub_acc):.2f}% "
        f"rows={len(rows)} n_eval={len(metric_rows)}\n  csv={csv_path}"
    )
    return summary


def build_cfg_from_args(args: argparse.Namespace) -> ExportConfig:
    if args.preset:
        base = PRESETS[args.preset]
        return ExportConfig(
            name=base.name,
            n_subs=base.n_subs,
            n_folds=base.n_folds,
            n_class=base.n_class,
            n_vids=base.n_vids,
            save_name=base.save_name,
            feat_dir=args.feat_dir or base.feat_dir,
            cp_dir=args.cp_dir or base.cp_dir,
            run=args.run if args.run is not None else base.run,
            mode=args.mode or base.mode,
            out_dir=args.out_dir or base.out_dir,
            subject_ids_pkl=base.subject_ids_pkl,
            time_len2=base.time_len2,
            time_step2=base.time_step2,
            trial_sec=base.trial_sec,
            ckpt_pick=base.ckpt_pick,
            use_allsubs_labels=base.use_allsubs_labels,
            ignore_invalid=base.ignore_invalid,
            class_names=base.class_names,
        )
    raise SystemExit(
        "Please pass --preset faced|ty9_movie|ty9_communication|"
        "ty9_movie_matched51|ty9_communication_matched51|"
        "ty8_movie_self|ty8_communication_self|"
        "faced_to_ty9_movie_mlp_ft|faced_to_ty9_communication_mlp_ft|"
        "ty9_movie_to_ty9_communication_mlp_ft_matched51"
    )


def main():
    parser = argparse.ArgumentParser(description="OOF per-window prediction export (no retrain)")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), required=True)
    parser.add_argument("--feat_dir", default=None)
    parser.add_argument("--cp_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--run", type=int, default=None)
    parser.add_argument("--mode", default=None, choices=["me", "de"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no_probs", action="store_true")
    args = parser.parse_args()

    cfg = build_cfg_from_args(args)
    device = resolve_device(args.device)
    print(
        f"preset={args.preset} device={device}\n"
        f"feat_dir={cfg.feat_dir}\ncp_dir={cfg.cp_dir}\nout_dir={cfg.out_dir}\n"
        f"run={cfg.run} folds={cfg.n_folds} n_subs={cfg.n_subs}"
    )
    export_one(cfg, device=device, write_probs=not args.no_probs)


if __name__ == "__main__":
    main()
