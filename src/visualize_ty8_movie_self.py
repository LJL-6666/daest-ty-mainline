#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TY8_movie_self 5 折分类结果可视化：被试准确率条形图 + 8 类混淆矩阵。
口径：52 人 / 28 视频 / 自评 argmax 8 类 / ME；读取 TY8_movie_self_ME_cp。

用法（在 FACED-base 目录下）:
  python visualize_ty8_movie_self.py
  python visualize_ty8_movie_self.py --feat_dir ... --cp_dir ... --out_dir ... --run 1
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import PDataset
from model.pl_models import MLPModel

N_SUBS = 52
N_FOLDS = 5
N_CLASS = 8
CHANCE_LEVEL = 100.0 / N_CLASS
SAVE_NAME = 'TY8_movie_self'
CLASS_NAMES = [
    'Anger',
    'Disgust',
    'Fear',
    'Sadness',
    'Amusement',
    'Inspiration',
    'Joy',
    'Tenderness',
]


def _has_required_feature_files(feat_dir: str, mode: str = 'me') -> bool:
    if not os.path.isdir(feat_dir):
        return False
    if not os.path.isfile(os.path.join(feat_dir, 'onesub_label2.npy')) and not os.path.isfile(
        os.path.join(feat_dir, 'allsubs_label2.npy')
    ):
        return False
    for fold in range(N_FOLDS):
        if not glob.glob(os.path.join(feat_dir, f'*_f{fold}_fea_{mode}.npy')):
            return False
    return True


def _has_required_mlp_ckpts(cp_dir: str, run: int) -> bool:
    run_dir = os.path.join(cp_dir, SAVE_NAME, f'r{run}')
    if not os.path.isdir(run_dir):
        return False
    for fold in range(N_FOLDS):
        ckpts = [p for p in glob.glob(os.path.join(run_dir, f'mlp_f{fold}_*.ckpt')) if '-v1' not in p]
        if not ckpts:
            return False
    return True


def _guess_default_base_roots():
    roots = []
    roots.append(os.path.join('runs', 'TY8_movie_self_ME_cp', 'runs'))
    extra = glob.glob(os.path.join('runs', 'TY8_movie_self_ME_cp*', 'runs'))
    extra = sorted(extra, key=lambda p: os.path.getmtime(p), reverse=True)
    for r in extra:
        if r not in roots:
            roots.append(r)
    return roots


def resolve_paths(feat_dir, cp_dir, out_dir, run=1, mode='me'):
    base_root = None
    if feat_dir is None or cp_dir is None:
        for root in _guess_default_base_roots():
            cand_feat = os.path.join(root, SAVE_NAME, 'ext_fea', f'fea_r{run}')
            cand_cp = os.path.join(root, 'cp')
            if _has_required_feature_files(cand_feat, mode=mode) and _has_required_mlp_ckpts(cand_cp, run):
                base_root = root
                if feat_dir is None:
                    feat_dir = cand_feat
                if cp_dir is None:
                    cp_dir = cand_cp
                break
        if feat_dir is None or cp_dir is None:
            raise FileNotFoundError(
                'Cannot locate TY8_movie_self features/checkpoints. '
                'Expected under runs/TY8_movie_self_ME_cp/runs/{TY8_movie_self/ext_fea/fea_r*, cp/TY8_movie_self/r*}'
            )
    if out_dir is None:
        if base_root is None:
            base_root = os.path.dirname(cp_dir.rstrip('/'))
        out_dir = os.path.join(base_root, 'visualization')
    return feat_dir, cp_dir, out_dir


def resolve_device(device: str) -> str:
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def get_val_subs_and_data(fold, feat_dir, mode='me'):
    n_per = round(N_SUBS / N_FOLDS)
    if fold < N_FOLDS - 1:
        val_subs = np.arange(n_per * fold, n_per * (fold + 1))
    else:
        val_subs = np.arange(n_per * fold, N_SUBS)
    pat = os.path.join(feat_dir, f'*_f{fold}_fea_{mode}.npy')
    files = glob.glob(pat)
    if not files:
        raise FileNotFoundError(f'No feature file for fold {fold}: {pat}')
    data2 = np.load(files[0])
    if np.isnan(data2).any():
        data2 = np.nan_to_num(data2, nan=0.0)
    fea_dim = data2.shape[-1]
    data2 = data2.reshape(N_SUBS, -1, fea_dim)
    allsubs_path = os.path.join(feat_dir, 'allsubs_label2.npy')
    if os.path.isfile(allsubs_path):
        allsubs_label2 = np.load(allsubs_path)
        if allsubs_label2.shape != (N_SUBS, data2.shape[1]):
            raise ValueError(
                f'allsubs_label2 shape={allsubs_label2.shape}，期望 ({N_SUBS}, {data2.shape[1]})'
            )
        uniq = sorted(set(int(x) for x in allsubs_label2.ravel().tolist()))
        labels_val = allsubs_label2[val_subs].reshape(-1)
        n_per_sub = allsubs_label2.shape[1]
    else:
        onesub_label2 = np.load(os.path.join(feat_dir, 'onesub_label2.npy'))
        uniq = sorted(set(int(x) for x in onesub_label2.tolist()))
        n_per_sub = len(onesub_label2)
        labels_val = np.tile(onesub_label2, len(val_subs))
        print('WARNING: allsubs_label2.npy missing; falling back to tiled onesub_label2')
    if min(uniq) < 0 or max(uniq) >= N_CLASS:
        # 允许 -1（无效全 0 视频）；其它负值或越界仍报错
        bad = [u for u in uniq if u != -1 and (u < 0 or u >= N_CLASS)]
        if bad or (min(uniq) < -1):
            raise ValueError(f'labels 类别越界: {uniq}，期望 [0, {N_CLASS - 1}] 或 -1')
    valid_uniq = [u for u in uniq if u >= 0]
    if len(valid_uniq) < N_CLASS:
        print(f'WARNING: valid labels 仅覆盖 {len(valid_uniq)}/{N_CLASS} 类: {valid_uniq}')
    if -1 in uniq:
        print(f'INFO: labels contain invalid(-1); will be ignored in metrics')
    val_data = data2[val_subs].reshape(-1, fea_dim)
    return val_subs, val_data, labels_val, n_per_sub


def run_collect_predictions(feat_dir, cp_dir, run=1, device='cuda', mode='me'):
    all_pred, all_true, all_subject_id = [], [], []
    subject_correct = np.zeros(N_SUBS)
    subject_total = np.zeros(N_SUBS)
    mlp_dir = os.path.join(cp_dir, SAVE_NAME, f'r{run}')

    for fold in range(N_FOLDS):
        val_subs, val_data, labels_val, n_per_sub = get_val_subs_and_data(fold, feat_dir, mode=mode)
        ckpt_pattern = os.path.join(mlp_dir, f'mlp_f{fold}_wd=*.ckpt')
        ckpts = [f for f in glob.glob(ckpt_pattern) if '-v1' not in f]
        if not ckpts:
            ckpts = [f for f in glob.glob(os.path.join(mlp_dir, f'mlp_f{fold}_*.ckpt')) if '-v1' not in f]
        if not ckpts:
            raise FileNotFoundError(f'No MLP checkpoint for fold {fold}: {ckpt_pattern}')
        ckpt_path = max(ckpts, key=os.path.getmtime)

        predictor = MLPModel.load_from_checkpoint(ckpt_path, map_location=device)
        predictor.eval()
        with torch.no_grad():
            probe = torch.zeros(1, val_data.shape[-1], device=device)
            out_dim = int(predictor(probe).shape[-1])
        if out_dim != N_CLASS:
            raise ValueError(f'MLP out_dim={out_dim}，期望 {N_CLASS}（ckpt={ckpt_path}）')

        dataset = PDataset(val_data, labels_val.astype(np.int64))
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
        preds_fold, trues_fold = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = predictor(x)
                preds_fold.append(logits.argmax(1).cpu().numpy())
                trues_fold.append(y.cpu().numpy())
        preds_fold = np.concatenate(preds_fold)
        trues_fold = np.concatenate(trues_fold)

        for i in range(len(preds_fold)):
            if int(trues_fold[i]) < 0:
                continue  # 全 0 无效标签：按视频剔除，不计入
            sub_idx = i // n_per_sub
            sub_id = int(val_subs[sub_idx])
            all_pred.append(preds_fold[i])
            all_true.append(trues_fold[i])
            all_subject_id.append(sub_id)
            subject_total[sub_id] += 1
            if preds_fold[i] == trues_fold[i]:
                subject_correct[sub_id] += 1

    return (
        np.array(all_pred),
        np.array(all_true),
        np.array(all_subject_id),
        subject_correct,
        subject_total,
    )


def plot_subject_accuracy_bar(acc_per_sub, out_path):
    valid_acc = acc_per_sub[np.isfinite(acc_per_sub)]
    n_unique = len(np.unique(np.round(valid_acc, 6))) if len(valid_acc) else 0
    sorted_acc = np.sort(valid_acc)
    mean_acc = float(np.mean(valid_acc)) if len(valid_acc) else float('nan')
    std_acc = float(np.std(valid_acc)) if len(valid_acc) else float('nan')

    fig, ax = plt.subplots(figsize=(12, 5))
    x_sub = np.arange(len(sorted_acc))
    ax.bar(x_sub, sorted_acc, color='lightgray', edgecolor='gray', linewidth=0.8, label='accuracy for each subject')
    x_avg = len(sorted_acc) + 4
    ax.bar(x_avg, mean_acc, color='lightgreen', edgecolor='green', linewidth=0.8, width=1.2, label='averaged accuracy')
    ax.errorbar(x_avg, mean_acc, yerr=std_acc, color='black', capsize=4, capthick=1, fmt='none')
    ax.text(x_avg, mean_acc + std_acc + 2, f'{mean_acc:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', linewidth=1.5, label=f'Chance level: {CHANCE_LEVEL:.2f}%')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlim(-0.5, len(sorted_acc) + 8)
    ax.set_ylim(0, 105)
    step = 10
    ax.set_xticks(list(range(0, len(sorted_acc), step)) + [len(sorted_acc), x_avg])
    ax.set_xticklabels([str(i) for i in range(0, len(sorted_acc), step)] + [str(len(sorted_acc)), 'Avg'])
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('TY8_movie_self 5-Fold Classification - Subject Accuracy')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(
        f'Saved: {out_path}  '
        f'(subject accuracy: {n_unique} unique values among {len(valid_acc)}/{N_SUBS} valid subjects)'
    )


def plot_confusion_matrix(all_pred, all_true, out_path, csv_path=None):
    cm = np.zeros((N_CLASS, N_CLASS), dtype=np.float64)
    for t, p in zip(all_true, all_pred):
        cm[int(t), int(p)] += 1
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    cm_pct = 100.0 * cm / row_sum

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(np.arange(N_CLASS))
    ax.set_yticks(np.arange(N_CLASS))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    for i in range(N_CLASS):
        for j in range(N_CLASS):
            ax.text(
                j,
                i,
                f'{cm_pct[i, j]:.1f}',
                ha='center',
                va='center',
                color='black' if cm_pct[i, j] < 50 else 'white',
                fontsize=9,
            )
    plt.colorbar(im, ax=ax, label='%')
    ax.set_title('TY8_movie_self 5-Fold Classification - Confusion Matrix (Row Norm.)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

    if csv_path is not None:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([''] + CLASS_NAMES)
            for i, name in enumerate(CLASS_NAMES):
                writer.writerow([name] + [f'{cm_pct[i, j]:.6f}' for j in range(N_CLASS)])
        print(f'Saved: {csv_path}')


def main():
    parser = argparse.ArgumentParser(description='TY8_movie_self 5-Fold 8-class visualization')
    parser.add_argument('--feat_dir', type=str, default=None)
    parser.add_argument('--cp_dir', type=str, default=None, help='Checkpoint root containing TY8_movie_self/r*/')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--mode', type=str, default='me', choices=['me', 'de'])
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()

    args.feat_dir, args.cp_dir, args.out_dir = resolve_paths(
        args.feat_dir, args.cp_dir, args.out_dir, run=args.run, mode=args.mode
    )
    args.device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Using feat_dir: {args.feat_dir}')
    print(f'Using cp_dir:   {args.cp_dir}')
    print(f'Using out_dir:  {args.out_dir}')
    print(f'Using device:   {args.device}')
    print(f'Using mode:     {args.mode}')
    print(f'N_SUBS={N_SUBS} N_FOLDS={N_FOLDS} N_CLASS={N_CLASS} chance={CHANCE_LEVEL:.2f}%')

    all_pred, all_true, _, subject_correct, subject_total = run_collect_predictions(
        args.feat_dir, args.cp_dir, run=args.run, device=args.device, mode=args.mode
    )
    acc_per_sub = np.array(
        [
            100.0 * subject_correct[i] / subject_total[i] if subject_total[i] > 0 else np.nan
            for i in range(N_SUBS)
        ]
    )
    plot_subject_accuracy_bar(
        acc_per_sub, os.path.join(args.out_dir, f'ty8_movie_self_5folds_subject_accuracy_r{args.run}.png')
    )
    plot_confusion_matrix(
        all_pred,
        all_true,
        os.path.join(args.out_dir, f'ty8_movie_self_5folds_cls8_confusion_r{args.run}.png'),
        csv_path=os.path.join(args.out_dir, f'ty8_movie_self_5folds_cls8_confusion_matrix_r{args.run}.csv'),
    )

    overall_acc = 100.0 * (all_pred == all_true).sum() / len(all_pred)
    print(f'\nOverall Accuracy: {overall_acc:.2f}%')
    print(f'Mean Subject Accuracy: {np.nanmean(acc_per_sub):.2f}% ± {np.nanstd(acc_per_sub):.2f}%')
    print(f'Valid Subjects: {int(np.isfinite(acc_per_sub).sum())}/{N_SUBS}')
    print(f'Chance Level: {CHANCE_LEVEL:.2f}%')


if __name__ == '__main__':
    main()
