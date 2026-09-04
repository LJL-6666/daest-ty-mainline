import re
import scipy.io as sio
from glob import glob
import hdf5storage
import numpy as np
import os
import copy

try:
    import pandas as pd
except ImportError:
    pd = None

# 6 分类 19 视频 ID（与 Clisa_6-all config_6class.SELECTED_19_VIDEOS 一致，用于问卷 CSV 顺序）
SELECTED_19_VIDEOS = [1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28]


def _resolve_questionnaire_subject_dir(questionnaire_dir, sub_id):
    questionnaire_dir = os.path.expanduser(questionnaire_dir)
    sub_dir = os.path.join(questionnaire_dir, str(sub_id))
    if not os.path.isdir(sub_dir) and str(sub_id).replace(' ', '').isdigit():
        sub_dir_alt = os.path.join(questionnaire_dir, str(int(str(sub_id).strip())))
        if os.path.isdir(sub_dir_alt):
            sub_dir = sub_dir_alt
    return sub_dir


def _read_questionnaire_rating_csv(sub_dir, task='exp1'):
    if task == 'exp1':
        pattern = os.path.join(sub_dir, 'exp1*rating*.csv')
    else:
        pattern = os.path.join(sub_dir, 'exp0*rating*.csv')
    files = sorted(glob(pattern))
    if not files or pd is None:
        return None
    df = pd.read_csv(files[0])
    if 'videoIndex' not in df.columns:
        return None
    return df['videoIndex'].values.astype(int)


def _video_order_from_questionnaire_csv(questionnaire_dir, subject_ids, task='exp1', n_vids=19):
    """从问卷 CSV 读取播放顺序。

    n_vids=19（TY6）：返回 playback_order[video_slot]=play_rank（与 Clisa_6-all 一致）。
    n_vids=28（TY9）：返回 vid_orders[play_pos]=videoIndex(1..28)，与 FACED After_remarks 一致。
    """
    if pd is None:
        return np.tile(np.arange(n_vids, dtype=np.int32), (len(subject_ids), 1))
    questionnaire_dir = os.path.expanduser(questionnaire_dir)
    vid_orders = np.zeros((len(subject_ids), n_vids), dtype=np.int32)
    for idx, sub_id in enumerate(subject_ids):
        sub_dir = _resolve_questionnaire_subject_dir(questionnaire_dir, sub_id)
        if not os.path.isdir(sub_dir):
            if n_vids == 28:
                vid_orders[idx, :] = np.arange(1, n_vids + 1, dtype=np.int32)
            else:
                vid_orders[idx, :] = np.arange(n_vids)
            continue
        try:
            all_video_indices = _read_questionnaire_rating_csv(sub_dir, task=task)
            if all_video_indices is None:
                raise ValueError('missing videoIndex')
            if n_vids == 28:
                vals = all_video_indices[:n_vids]
                if len(vals) != n_vids:
                    raise ValueError(f'expected {n_vids} rows, got {len(vals)}')
                expected = set(range(1, n_vids + 1))
                if set(int(v) for v in vals) != expected:
                    raise ValueError('videoIndex is not a permutation of 1..28')
                vid_orders[idx, :] = vals
                continue
            playback_positions = []
            for play_pos, video_num in enumerate(all_video_indices):
                if int(video_num) in SELECTED_19_VIDEOS:
                    playback_positions.append((play_pos, int(video_num)))
            if len(playback_positions) != n_vids:
                raise ValueError('selected 19 videos mismatch')
            playback_positions.sort(key=lambda x: x[0])
            video_num_to_new_idx = {v: i for i, v in enumerate(SELECTED_19_VIDEOS)}
            playback_order = np.zeros(n_vids, dtype=np.int32)
            for relative_pos, (_, video_num) in enumerate(playback_positions):
                new_idx = video_num_to_new_idx[video_num]
                playback_order[new_idx] = relative_pos
            vid_orders[idx, :] = playback_order
        except Exception:
            if n_vids == 28:
                vid_orders[idx, :] = np.arange(1, n_vids + 1, dtype=np.int32)
            else:
                vid_orders[idx, :] = np.arange(n_vids)
    return vid_orders


def ty_pkl_playback_to_video_index(data_all, vid_order):
    """TY 预处理 pkl 第一维为播放顺序；转为 videoIndex 槽位顺序（slot i = video i+1）。"""
    n_subs, n_vids = data_all.shape[0], data_all.shape[1]
    out = np.empty_like(data_all)
    for sub in range(n_subs):
        for play_pos in range(n_vids):
            vid_idx = int(vid_order[sub, play_pos]) - 1
            if vid_idx < 0 or vid_idx >= n_vids:
                raise ValueError(
                    f'subject row {sub}: invalid videoIndex {vid_order[sub, play_pos]} at play_pos {play_pos}'
                )
            out[sub, vid_idx] = data_all[sub, play_pos]
    return out


def ty_subject_video_playback_to_video_index(subject_video, vid_order):
    """逐被试标签矩阵与 EEG 同步：播放序 -> videoIndex 序。"""
    n_subs, n_vids = subject_video.shape
    out = np.empty_like(subject_video)
    for sub in range(n_subs):
        for play_pos in range(n_vids):
            vid_idx = int(vid_order[sub, play_pos]) - 1
            out[sub, vid_idx] = subject_video[sub, play_pos]
    return out


def ty_subject_ids_from_pkl(raw, n_subs):
    subject_ids = raw.get('subject_ids', [])
    if isinstance(subject_ids, np.ndarray):
        subject_ids = subject_ids.tolist()
    subject_ids = list(subject_ids)[:n_subs]
    if len(subject_ids) < n_subs:
        subject_ids = subject_ids + [str(i) for i in range(len(subject_ids), n_subs)]
    return subject_ids


def video_order_load(n_vids=28, dataset=None, questionnaire_dir=None, subject_ids=None, questionnaire_task=None, after_remarks_dir=None):
    """加载视频播放顺序。

    TY 问卷：n_vids in (19, 28) 且给出 questionnaire_dir 与 subject_ids 时从 CSV 读取。
    questionnaire_task: exp1=交流, exp0=电影。
    FACED 用 After_remarks：after_remarks_dir 可显式传入；否则用环境变量 FACED_AFTER_REMARKS_DIR；否则用脚本所在项目下的 After_remarks。
    """
    if n_vids in (19, 28) and questionnaire_dir is not None and subject_ids is not None:
        task = questionnaire_task if questionnaire_task is not None else ('exp1' if (dataset in ('first', 'second', 'both', 'TY6') or dataset is None) else 'exp0')
        return _video_order_from_questionnaire_csv(questionnaire_dir, subject_ids, task=task, n_vids=n_vids)
    if after_remarks_dir is None:
        after_remarks_dir = os.environ.get('FACED_AFTER_REMARKS_DIR', '')
    if not after_remarks_dir or not os.path.isdir(after_remarks_dir):
        # 默认：脚本在 FACED-base/utils/reorder_vids.py，项目根为 ../
        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        after_remarks_dir = os.path.join(_base, 'After_remarks')
    if not os.path.isdir(after_remarks_dir):
        raise FileNotFoundError(
            f"FACED 视频顺序需要 After_remarks 目录，未找到: {after_remarks_dir}。"
            "请设置 data.after_remarks_dir 或环境变量 FACED_AFTER_REMARKS_DIR。"
        )
    datapath = after_remarks_dir
    filesPath = os.listdir(datapath)
    # 只保留子目录并按被试编号排序，与 processed_data 的 sub000..sub122 顺序一致
    filesPath = sorted([f for f in filesPath if os.path.isdir(os.path.join(datapath, f))],
                      key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
    vid_orders = np.zeros((len(filesPath), n_vids), dtype=int)
    for idx, file in enumerate(filesPath):
        remark_file = os.path.join(datapath, file, 'After_remarks.mat')
        if not os.path.isfile(remark_file):
            raise FileNotFoundError(f"缺少视频顺序文件: {remark_file}")
        subject_remark = hdf5storage.loadmat(remark_file)['After_remark']
        vid_orders[idx, :] = [np.squeeze(subject_remark[vid][0][2]) for vid in range(0, n_vids)]
    return vid_orders


def reorder_vids(data, n_vids, vid_play_order):
    # data: (n_subs, n_points, n_feas)
    # return:(n_subs, n_points, n_feas)
    n_subs = data.shape[0]
    n_samples = data.shape[1]//n_vids
    vid_play_order_copy = vid_play_order.copy()
    if n_vids == 24:
        vid_play_order_new = np.zeros((n_subs, n_vids)).astype(np.int32)
        data_reorder = np.zeros_like(data)
        for sub in range(n_subs):
            tmp = vid_play_order_copy[sub,:]
            tmp = tmp[(tmp<13)|(tmp>16)]
            tmp[tmp>=17] = tmp[tmp>=17] - 4
            tmp = tmp - 1
            vid_play_order_new[sub, :] = tmp

            data_sub = data[sub, :, :]
            data_sub = data_sub.reshape(n_vids, n_samples, data_sub.shape[-1])
            data_sub = data_sub[tmp, :, :]
            data_reorder[sub, :, :] = data_sub.reshape(n_vids*n_samples, data_sub.shape[-1])
    elif n_vids == 28:
        vid_play_order_new = np.zeros((n_subs, n_vids)).astype(np.int32)
        data_reorder = np.zeros_like(data)
        for sub in range(n_subs):
            tmp = vid_play_order_copy[sub,:]
            tmp = tmp - 1
            vid_play_order_new[sub, :] = tmp

            data_sub = data[sub, :, :]
            data_sub = data_sub.reshape(n_vids, n_samples, data_sub.shape[-1])
            # Error occurs saying that the elements of tmp is not int
            tmp = [int(i) for i in tmp]
            data_sub = data_sub[tmp, :, :]
            data_reorder[sub, :, :] = data_sub.reshape(n_vids*n_samples, data_sub.shape[-1])
    return data_reorder, vid_play_order_new

def reorder_vids_sepVideo(data, vid_play_order, sel_vid_inds, n_vids_all):
    # data: (n_subs, n_points, n_feas)
    n_vids = len(sel_vid_inds)
    n_subs = data.shape[0]
    # print('n_subs:',n_subs)
    vid_play_order_copy = vid_play_order.copy()
    vid_play_order_new = np.zeros((n_subs, len(sel_vid_inds))).astype(np.int32)
    data_reorder = np.zeros_like(data)
    if n_vids_all == 24:
        for sub in range(n_subs):
            tmp = vid_play_order_copy[sub,:]
            tmp = tmp[(tmp<13)|(tmp>16)]
            tmp[tmp>=17] = tmp[tmp>=17] - 4
            tmp = tmp - 1

            tmp_new = []
            for i in range(len(tmp)):
                if tmp[i] in sel_vid_inds:
                    tmp_new.append(np.where(sel_vid_inds==tmp[i])[0][0])
            tmp_new = np.array(tmp_new)

            vid_play_order_new[sub, :] = tmp_new

            data_sub = data[sub, :, :]
            data_sub = data_sub.reshape(n_vids, -1, data_sub.shape[-1])
            data_sub = data_sub[tmp_new, :, :]
            data_reorder[sub, :, :] = data_sub.reshape(-1, data_sub.shape[-1])
    elif n_vids_all == 28:
        for sub in range(n_subs):
            tmp = vid_play_order_copy[sub,:]
            tmp = tmp - 1

            tmp_new = []
            for i in range(len(tmp)):
                if tmp[i] in sel_vid_inds:
                    tmp_new.append(np.where(sel_vid_inds==tmp[i])[0][0])
            tmp_new = np.array(tmp_new)

            vid_play_order_new[sub, :] = tmp_new

            data_sub = data[sub, :, :]
            data_sub = data_sub.reshape(n_vids, -1, data_sub.shape[-1])
            data_sub = data_sub[tmp_new, :, :]
            data_reorder[sub, :, :] = data_sub.reshape(-1, data_sub.shape[-1])
    elif n_vids_all == 19:
        # TY6：vid_play_order[sub, video_idx]=play_rank，按播放顺序重排（与 Clisa_6-all 一致）
        n_samples = data.shape[1] // 19
        for sub in range(n_subs):
            tmp = np.argsort(vid_play_order_copy[sub, :])
            vid_play_order_new[sub, :] = vid_play_order_copy[sub, :]
            data_sub = data[sub, :, :].reshape(19, n_samples, data.shape[-1])
            data_sub = data_sub[tmp, :, :]
            data_reorder[sub, :, :] = data_sub.reshape(-1, data_sub.shape[-1])
    return data_reorder, vid_play_order_new



def reorder_vids_back(data, n_vids, vid_play_order_new):
    # data: (n_subs, n_points, n_feas)
    # return:(n_subs, n_points, n_feas)
    n_subs = data.shape[0]
    n_samples = data.shape[1] // n_vids
    data_back = np.zeros((n_subs, n_vids, n_samples, data.shape[-1]))
    for sub in range(n_subs):
        data_sub = data[sub, :, :].reshape(n_vids, n_samples, data.shape[-1])
        if n_vids == 19:
            inv = np.argsort(vid_play_order_new[sub, :])
            data_back[sub, inv, :, :] = data_sub
        else:
            data_back[sub, vid_play_order_new[sub, :], :, :] = data_sub
    data_back = data_back.reshape(n_subs, n_vids * n_samples, data.shape[-1])
    return data_back


if __name__ == '__main__':
    video_order_load(28)
