# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
import gc
from tqdm.auto import tqdm
from joblib import Parallel, delayed


# 分类特征
cat_features = [
    "B_30", "B_38", "D_114", "D_116", "D_117",
    "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"
]


def get_difference(df, num_features):
    """计算每个 customer 的最后一行与倒数第二行数值特征的差值"""
    def process_customer_diff(customer_id, customer_df):
        diff = customer_df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        return customer_id, diff

    print("Computing difference features with tqdm and joblib...")
    customer_groups = list(df.groupby("customer_ID"))
    results = Parallel(n_jobs=-1)(
        delayed(process_customer_diff)(cid, group)
        for cid, group in tqdm(customer_groups, desc="Diff features")
    )

    customer_ids, diffs = zip(*results)
    diff_df = pd.DataFrame(np.concatenate(diffs, axis=0), 
                           columns=[f"{col}_diff1" for col in num_features])
    diff_df["customer_ID"] = customer_ids
    return diff_df


def process_features(df):
    # 提取特征列
    features = df.drop(columns=["customer_ID", "S_2", "target"]).columns.tolist()
    num_features = [col for col in features if col not in cat_features]

    print("Processing numerical aggregations...")
    tqdm.pandas(desc="Num agg")
    num_agg = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    num_agg.columns = ['_'.join(col) for col in num_agg.columns]
    num_agg.reset_index(inplace=True)

    print("Processing categorical aggregations...")
    tqdm.pandas(desc="Cat agg")
    cat_agg = df.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
    cat_agg.columns = ['_'.join(col) for col in cat_agg.columns]
    cat_agg.reset_index(inplace=True)

    print("Computing difference features...")
    diff_df = get_difference(df, num_features)

    print("Merging all features...")
    full_df = num_agg.merge(cat_agg, on="customer_ID", how="inner")\
                     .merge(diff_df, on="customer_ID", how="inner")

    # 加入标签
    target_df = df[['customer_ID', 'target']].drop_duplicates()
    full_df = full_df.merge(target_df, on="customer_ID", how="inner")

    print("Optimizing data types...")
    for col in tqdm(full_df.select_dtypes(include='float64').columns, desc="Float64 → Float32"):
        full_df[col] = full_df[col].astype(np.float32)
    for col in tqdm(full_df.select_dtypes(include='int64').columns, desc="Int64 → Int32"):
        full_df[col] = full_df[col].astype(np.int32)

    gc.collect()
    return full_df
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    INPUT_PATH = "./1_sample_train_with_labels.csv"
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    processed_df = process_features(df)
    processed_df.to_csv('./features.csv', index=False, encoding="utf-8-sig")
