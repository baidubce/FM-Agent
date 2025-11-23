"""Feature Mining Evaluator."""
import os
import json
import gc
import pandas as pd
import numpy as np
import random
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from filelock import FileLock
import tempfile
import subprocess
import shutil
import warnings
import datetime

warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ========== Config ==========
class CFG:
    seed = 42
    n_folds = 5
    target = 'target'
    beam_size = 3         # beam宽度
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    feature_dir = os.path.join(BASE_DIR, 'new_features')
    py_dir = os.path.join(BASE_DIR, 'new_py')
    beam_path = os.path.join(BASE_DIR, 'beam.json')
 

os.makedirs(CFG.feature_dir, exist_ok=True)
os.makedirs(CFG.py_dir, exist_ok=True)
 

# ========== 工具函数 ==========
def seed_everything():
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    os.environ['PYTHONHASHSEED'] = str(CFG.seed)

    
def amex_metric(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1]/gini[0] + top_four)
 

def xgb_amex_metric(preds, dtrain):
    labels = dtrain.get_label()
    return 'amex_metric', amex_metric(labels, preds)
 

def calculate_metrics(y_true, y_pred, threshold=0.5):
    y_pred_label = (y_pred >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_label)
    return {
        "AUC": auc,
        "F1": f1,
    }
 

def load_beam(beam_path):
    """ 读取 beam.json，不加锁，只读 """
    if os.path.exists(beam_path):
        with open(beam_path, 'r') as f:
            return json.load(f)
    else:
        return {"beam_size": CFG.beam_size, "paths": []}
 

def save_beam(beam, beam_path):
    """ 写入 beam.json（需要外部加锁） """
    with open(beam_path, 'w') as f:
        json.dump(beam, f, indent=4)

 
def merge_features(feature_list):
    """
    合并一组特征 csv 文件（每个都含 customer_ID/target），
    直接拼接所有特征，无需再 merge df_base
    """
    if not feature_list:
        raise ValueError("特征列表为空，无法合并！")
    dfs = [pd.read_csv(os.path.join(CFG.feature_dir, f)) for f in feature_list]
    # 先用第一个表为主，依次 left join 其它表
    df_merged = dfs[0]
    for df in dfs[1:]:
        overlap_cols = [col for col in df_merged.columns if col in df.columns and col not in ['customer_ID', 'target']]
        df_merged = df_merged.drop(columns=overlap_cols)
        df_merged = pd.merge(df_merged, df, on=['customer_ID', 'target'], how='left')
    return df_merged
 

def add_features(df_new, df_old):
    overlap_cols = [col for col in df_new.columns if col in df_old.columns and col not in ['customer_ID', 'target']]
    df_new = df_new.drop(columns=overlap_cols)
    df_output = pd.merge(df_new, df_old, on=['customer_ID', 'target'], how='left')
    return df_output
 

def train_and_cv(train):
    seed_everything()
 
    features = [col for col in train.columns if col not in ['customer_ID', CFG.target]]
    train.replace(np.inf, 1e18, inplace=True)
    train.replace(-np.inf, -1e18, inplace=True)
 
    oof_predictions = np.zeros(len(train))
    kfold = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    metrics_record = []
    importance_list = []
 
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(train, train[CFG.target])):
        x_train = train[features].iloc[trn_idx]
        y_train = train[CFG.target].iloc[trn_idx]
        x_val = train[features].iloc[val_idx]
        y_val = train[CFG.target].iloc[val_idx]
 
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
 
        evals = [(dtrain, 'train'), (dval, 'valid')]
        model = xgb.train(
            {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                # 'device': f'cuda:{gpu_id}',
                'seed': CFG.seed,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.2,
                'lambda': 1,
                'min_child_weight': 40,
                'n_jobs': -1,
                'verbosity': 1,
                'gamma': 0.5,
                'max_leaves': 100,
            },
            dtrain,
            num_boost_round=2000,
            evals=evals,
            early_stopping_rounds=200,
            # feval=xgb_amex_metric,
            maximize=True,
            verbose_eval=False
        )
 
        val_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        oof_predictions[val_idx] = val_pred
 
        metric_result = calculate_metrics(y_val.values, val_pred)
        metric_result["AMEX"] = amex_metric(y_val.values, val_pred)
        metrics_record.append(metric_result)
 
        importance = model.get_score(importance_type='gain')
        importance_list.append(importance)
 
        del x_train, y_train, x_val, y_val, dtrain, dval
        gc.collect()
 
    # 平均 importance
    importance_df = pd.DataFrame(importance_list).fillna(0)
    importance_avg_series = importance_df.mean()
    # 可选：去掉所有 importance 为 0 的列
    importance_avg_series = importance_avg_series
    importance_sorted = importance_avg_series.sort_values(ascending=False)
    importance_dict = {
        "top_20": list(importance_sorted.head(20).index),
        "bottom_20": list(importance_sorted.tail(20).index)
    }
 
    overall_score = amex_metric(train[CFG.target], oof_predictions)
    metrics_df = pd.DataFrame(metrics_record)
    metrics_avg = {col: metrics_df[col].mean() for col in metrics_df.columns}
    metrics_avg["AMEX"] = overall_score
    metrics_avg["avg_feature_importance"] = importance_dict
    return metrics_avg
 

def calculate_feature_overlap_rate(df_feat, df_old):
    """
    计算 df_feat 中的特征列与 df_old 的重复率。
    重复率 = 交集列数 / df_feat 的特征列数
    忽略 'customer_ID' 和 'target' 列。
 
    返回：
        overlap_rate (float): 重复率，范围 [0,1]
        overlap_cols (list): 重复的列名列表
    """
    # 类型检查
    if not isinstance(df_feat, pd.DataFrame) or not isinstance(df_old, pd.DataFrame):
        raise ValueError("输入必须是两个 pandas DataFrame")
 
    exclude_cols = {'customer_ID', 'target'}
 
    try:
        feat_cols = set(df_feat.columns) - exclude_cols
        old_cols = set(df_old.columns) - exclude_cols
 
        # 只保留字符串类型列名
        feat_cols = {col for col in feat_cols if isinstance(col, str)}
        old_cols = {col for col in old_cols if isinstance(col, str)}
 
        if not feat_cols:
            return 0.0, []
 
        overlap_cols = feat_cols & old_cols
        overlap_rate = len(overlap_cols) / len(feat_cols)
 
        return overlap_rate, list(overlap_cols)
 
    except Exception as e:
        # 捕获任何异常，返回0重复率，附带错误信息打印
        print(f"⚠️ 计算重复率失败: {str(e)}")
        return 0.0, []
 
 
def execute_py_and_get_csv(
    py_path, feature_file_name='features.csv'):
    """
    执行特征工程py脚本，生成csv到 tmp_dir ，并返回其路径
    """
    tmp_csv = feature_file_name
    cmd = [
       "python", py_path,
        "--out_path", tmp_csv
    ]
    print("tmp_csv:", tmp_csv)
    proc = subprocess.Popen(
        cmd,
        # cwd=tmp_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8"
    )
    out, err = proc.communicate(timeout=7200)
    if proc.returncode != 0:
        raise RuntimeError(f"特征脚本运行失败:\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    if not os.path.exists(tmp_csv):
        raise RuntimeError("特征脚本未生成指定csv文件")
    return tmp_csv
 

def evaluate(path_user_py):
    """
    输入新特征py路径，自动评测临时csv。
    返回dict: 包含评分和详细error_info
    """
    result = {
        "combined_score":0,
        "AMEX": None,
        "AUC": None,
        "F1": None,
        "num_new_features": None,
        "avg_feature_importance": None,
        "error_info": {
        }
    }
 
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            py_path = path_user_py
            # 1. 执行特征py，生成csv
            try:
                tmp_csv = execute_py_and_get_csv(py_path)
            except Exception as e:
                result["error_info"]["run_error"] = str(e)
                return result
 
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            csv_name = f"new_{timestamp}.csv"
            py_name = f"new_{timestamp}.py"
 
            # 2. 检查df_feat
            try:
                df_feat = pd.read_csv(tmp_csv)
                result["num_new_features"] = df_feat.shape[1] - 2
                # 这里可根据实际加行列数的检查
                # if df_feat.shape[0] != ...: raise ValueError("行数不符")
            except Exception as e:
                result["error_info"]["data_error"] = f"读取csv失败: {str(e)}"
                return result
 
            # 3. beam初始化
            try:
                beam = load_beam(CFG.beam_path)
            except Exception as e:
                result["error_info"]["evaluate_error"] = f"加载beam失败: {str(e)}"
                return result
 
            if not beam['paths']:
                print("============== beam初始化 ==============")
                try:
                    metrics = train_and_cv(df_feat)
                    result["AMEX"] = metrics.get('AMEX')
                    result["AUC"] = metrics.get('AUC')
                    result["F1"] = metrics.get('F1')
                    result["avg_feature_importance"] = metrics.get("avg_feature_importance")
                    result["combined_score"] = metrics.get('AMEX')
 
                except Exception as e:
                    result["error_info"]["evaluate_error"] = f"训练评测失败: {str(e)}"
                    return result
                beam['paths'] = [{
                    'features': [csv_name],
                    'scripts': [py_name],
                    'metric': metrics['AMEX'],
                    'history': [{'script': py_name, 'metric': metrics['AMEX']}]
                }]
                shutil.copy(tmp_csv, os.path.join(CFG.feature_dir, csv_name))
                shutil.copy(py_path, os.path.join(CFG.py_dir, py_name))
                save_beam(beam, CFG.beam_path)
                print(f"beam初始化：已保存{csv_name}, {py_name}")
                return result
 
            # 4. 遍历beam路径合并评测
            candidates = []
            max_metrics = None
            for path in beam['paths']:
                feat_list = [f for f in path['features']]
                print(f"当前测评文件:{feat_list}")
                df_old = merge_features(feat_list)
                df_merged = add_features(df_feat,df_old)
 
                # 通过名称检测特征重复
                overlap_rate, overlap_cols = calculate_feature_overlap_rate(df_feat, df_old)
                print(f"features overlap_rate:{overlap_rate}")
                if overlap_rate > 0.5:
                    result["error_info"]["evaluate_error"] = f"新特征与历史特征有重叠: {overlap_cols}, 考虑更换输入特征或改变处理方式"
                    return result
 
 
                # 模型训练
                try:
                    print("开始测评新特征")
                    metrics = train_and_cv(df_merged)
                    print(metrics)
                except Exception as e:
                    result["error_info"]["evaluate_error"] = f"合并/训练出错: {str(e)}"
                    continue
 
                # 记录最高分
                if (not max_metrics) or (metrics['AMEX'] > max_metrics['AMEX']):
                    max_metrics = metrics
 
                new_candidate = {
                    'features': path['features'] + [csv_name],
                    'scripts': path['scripts'] + [py_name],
                    'metric': metrics['AMEX'],
                    'history': path['history'] + [{'script': py_name, 'metric': metrics['AMEX']}]
                }
                candidates.append(new_candidate)
 
            if max_metrics:
                result["AMEX"] = max_metrics.get('AMEX')
                result["AUC"] = max_metrics.get('AUC')
                result["F1"] = max_metrics.get('F1')
                result["avg_feature_importance"] = max_metrics.get("avg_feature_importance")
                result["combined_score"] = max_metrics.get('AMEX')
            # 5. 加锁，筛选topK beam，保存新入选文件
            lock_path = CFG.beam_path + ".lock"
            try:
                with FileLock(lock_path, timeout=60):
                    beam = load_beam(CFG.beam_path)
                    all_candidates = beam['paths'] + candidates
                    all_candidates = sorted(all_candidates, key=lambda x: x['metric'], reverse=True)
                    next_beam = all_candidates[:CFG.beam_size]
                    updated = False
                    for cand in candidates:
                        if cand in next_beam and cand not in beam['paths']:
                            shutil.copy(tmp_csv, os.path.join(CFG.feature_dir, csv_name))
                            shutil.copy(py_path, os.path.join(CFG.py_dir, py_name))
                            print(f"新特征入选beam，已保存到: {csv_name}, {py_name}")
                            updated = True
                    if updated:
                        beam['paths'] = next_beam
                        save_beam(beam, CFG.beam_path)
                        print("beam已更新")
                    else:
                        print("新特征未进入beam，无需保存文件。")
            except Exception as e:
                result["error_info"]["evaluate_error"] = f"beam加锁/保存失败: {str(e)}"
    except Exception as e:
        result["error_info"]["evaluate_error"] = f"未知异常: {str(e)}"
    return result
 

if __name__ == "__main__":
    # 评测单个特征工程脚本
    result = evaluate("./init.py")
    print(result)
