#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FACED_05_47 10 折分类结果可视化：被试准确率条形图 + 9 类混淆矩阵。
与参考代码一致：条形图无抖动。
用法（在 FACED-base 目录下）:
  python visualize_faced_05_47.py
  python visualize_faced_05_47.py --feat_dir ... --cp_dir ... --out_dir ...
"""
import argparse
import glob
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.dataset import PDataset
from model.pl_models import MLPModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os as _os
RUNS_ROOT = _os.environ.get("DAEST_RUNS_ROOT", ".")
RIEM_ROOT = _os.environ.get("DAEST_RIEM_ROOT", RUNS_ROOT)
PREP_ROOT = _os.environ.get("DAEST_PREP_ROOT", ".")
DATA_ROOT = _os.environ.get("DAEST_DATA_ROOT", ".")


N_SUBS = 123
N_FOLDS = 10
N_CLASS = 9
CHANCE_LEVEL = 100.0 / N_CLASS


def get_val_subs_and_data(fold, feat_dir):
    """返回该折的验证被试、验证特征、标签、每被试样本数。"""
    n_per = round(N_SUBS / N_FOLDS)
    if fold < N_FOLDS - 1:
        val_subs = np.arange(n_per * fold, n_per * (fold + 1))
    else:
        val_subs = np.arange(n_per * fold, N_SUBS)
    pat = os.path.join(feat_dir, f'*_f{fold}_fea_me.npy')
    files = glob.glob(pat)
    if not files:
        raise FileNotFoundError(f'No feature file for fold {fold}: {pat}')
    data2 = np.load(files[0])
    if np.isnan(data2).any():
        data2 = np.nan_to_num(data2, nan=0.0)
    onesub_label2 = np.load(os.path.join(feat_dir, 'onesub_label2.npy'))
    n_per_sub = len(onesub_label2)
    fea_dim = data2.shape[-1]
    data2 = data2.reshape(N_SUBS, -1, fea_dim)
    val_data = data2[val_subs].reshape(-1, fea_dim)
    labels_val = np.tile(onesub_label2, len(val_subs))
    return val_subs, val_data, labels_val, n_per_sub


def run_collect_predictions(feat_dir, cp_dir, run=1, device='cuda'):
    """加载每折 MLP，在验证集上推理，收集 (pred, true, subject_id) 及被试正确数/总数。"""
    all_pred, all_true, all_subject_id = [], [], []
    subject_correct = np.zeros(N_SUBS)
    subject_total = np.zeros(N_SUBS)
    mlp_dir = os.path.join(cp_dir, 'FACED', f'r{run}')

    for fold in range(N_FOLDS):
        val_subs, val_data, labels_val, n_per_sub = get_val_subs_and_data(fold, feat_dir)
        ckpt_pattern = os.path.join(mlp_dir, f'mlp_f{fold}_wd=*.ckpt')
        ckpts = [f for f in glob.glob(ckpt_pattern) if '-v1' not in f]
        if not ckpts:
            ckpts = glob.glob(os.path.join(mlp_dir, f'mlp_f{fold}_*.ckpt'))
            ckpts = [f for f in ckpts if '-v1' not in f]
        if not ckpts:
            raise FileNotFoundError(f'No MLP checkpoint for fold {fold}: {ckpt_pattern}')
        ckpt_path = ckpts[0]

        predictor = MLPModel.load_from_checkpoint(ckpt_path, map_location=device)
        predictor.eval()
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
            sub_idx = i // n_per_sub
            sub_id = val_subs[sub_idx]
            all_pred.append(preds_fold[i])
            all_true.append(trues_fold[i])
            all_subject_id.append(sub_id)
            subject_total[sub_id] += 1
            if preds_fold[i] == trues_fold[i]:
                subject_correct[sub_id] += 1

    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    all_subject_id = np.array(all_subject_id)
    return all_pred, all_true, all_subject_id, subject_correct, subject_total


def plot_subject_accuracy_bar(acc_per_sub, out_path):
    """被试准确率条形图：灰条升序 + 平均绿条 + 误差条 + 随机水平线（与参考代码一致，无抖动）。"""
    n_unique = len(np.unique(np.round(acc_per_sub, 6)))
    sorted_acc = np.sort(acc_per_sub)
    mean_acc = np.mean(acc_per_sub)
    std_acc = np.std(acc_per_sub)

    fig, ax = plt.subplots(figsize=(14, 5))
    x_sub = np.arange(N_SUBS)
    ax.bar(x_sub, sorted_acc, color='lightgray', edgecolor='gray', linewidth=0.8, label='accuracy for each subject')
    x_avg = N_SUBS + 4
    ax.bar(x_avg, mean_acc, color='lightgreen', edgecolor='green', linewidth=0.8, width=1.2, label='averaged accuracy')
    ax.errorbar(x_avg, mean_acc, yerr=std_acc, color='black', capsize=4, capthick=1, fmt='none')
    ax.text(x_avg, mean_acc + std_acc + 2, f'{mean_acc:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', linewidth=1.5, label=f'Chance level: {CHANCE_LEVEL:.2f}%')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlim(-0.5, N_SUBS + 8)
    ax.set_ylim(0, 105)
    step = 20 if N_SUBS > 50 else 10
    ax.set_xticks(list(range(0, N_SUBS, step)) + [N_SUBS, x_avg])
    ax.set_xticklabels([str(i) for i in range(0, N_SUBS, step)] + [str(N_SUBS), 'Avg'])
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('FACED_05_47 10-Fold Classification - Subject Accuracy')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}  (subject accuracy: {n_unique} unique values among {N_SUBS} subjects)')


def plot_confusion_matrix(all_pred, all_true, out_path):
    """9 类混淆矩阵热力图：行为 True Label，列为 Predicted Label，单元格为百分比。"""
    cm = np.zeros((N_CLASS, N_CLASS), dtype=np.float64)
    for t, p in zip(all_true, all_pred):
        cm[int(t), int(p)] += 1
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    cm_pct = 100.0 * cm / row_sum
    class_names = ['Anger', 'Disgust', 'Fear', 'Sadness', 'neutral', 'Amusement', 'Inspiration', 'Joy', 'Tenderness']
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(np.arange(N_CLASS))
    ax.set_yticks(np.arange(N_CLASS))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    for i in range(N_CLASS):
        for j in range(N_CLASS):
            ax.text(j, i, f'{cm_pct[i, j]:.1f}', ha='center', va='center', 
                   color='black' if cm_pct[i, j] < 50 else 'white', fontsize=9)
    plt.colorbar(im, ax=ax, label='%')
    ax.set_title('FACED_05_47 10-Fold Classification - Confusion Matrix (Row Norm.)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='FACED_05_47 10-Fold visualization')
    parser.add_argument('--feat_dir', type=str, 
                       default=f'{DATA_ROOT}/data-faced/0.05–47 Hz_mat/ext_fea/fea_r1',
                       help='Path to folder containing *_f*_fea_me.npy and onesub_label2.npy')
    parser.add_argument('--cp_dir', type=str, 
                       default='runs/FACED_05_47_cp/runs/cp',
                       help='Checkpoint root containing FACED/r1/ with mlp_f*.ckpt')
    parser.add_argument('--out_dir', type=str, 
                       default='runs/FACED_05_47_cp/visualization',
                       help='Output directory for figures')
    parser.add_argument('--run', type=int, default=1, help='Run id (r1, r2, ...)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_pred, all_true, _, subject_correct, subject_total = run_collect_predictions(
        args.feat_dir, args.cp_dir, run=args.run, device=args.device)
    acc_per_sub = np.array([100.0 * subject_correct[i] / subject_total[i] if subject_total[i] > 0 else 0.0 for i in range(N_SUBS)])
    plot_subject_accuracy_bar(acc_per_sub, os.path.join(args.out_dir, 'faced_05_47_10folds_subject_accuracy.png'))
    plot_confusion_matrix(all_pred, all_true, os.path.join(args.out_dir, 'faced_05_47_10folds_cls9_confusion.png'))
    
    # 打印总体统计信息
    overall_acc = 100.0 * (all_pred == all_true).sum() / len(all_pred)
    print(f'\nOverall Accuracy: {overall_acc:.2f}%')
    print(f'Mean Subject Accuracy: {np.mean(acc_per_sub):.2f}% ± {np.std(acc_per_sub):.2f}%')
    print(f'Chance Level: {CHANCE_LEVEL:.2f}%')


if __name__ == '__main__':
    main()

