import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import json
import os
from collections import Counter


def iloc_list(samples, indexes):
    new_samples = []
    for indx in indexes:
        new_samples.append(samples[indx])
    return new_samples


samples = json.load(open("../data/reddit/reddit_data.json", encoding="utf-8"))
subreddits_label = ["android", "apple", "nba", "movies", "playstation", "technology", "dota2"]
doc_ids = list(range(1, len(samples) + 1))

# ========== 关键修改：按板块分组并保留前15个样本 ==========
# 1. 初始化字典，按subreddit分组存储样本
subreddit_samples = {label: [] for label in subreddits_label}
# 2. 遍历原始样本，按板块归类
for sample in samples:
    subreddit = sample["subreddit"]
    if subreddit in subreddits_label:
        # 只添加到对应板块列表，暂不合并
        subreddit_samples[subreddit].append(sample)
# 3. 每个板块仅保留前15个样本，合并成最终样本集
new_samples = []
subreddits = []  # 对应new_samples的板块标签（用于后续分层划分）
for label in subreddits_label:
    # 截取前15个，不足15则保留全部
    top15_samples = subreddit_samples[label][:15]
    new_samples.extend(top15_samples)
    # 同步更新subreddits列表（每个样本对应一个板块标签）
    subreddits.extend([label] * len(top15_samples))
# ========== 结束修改 ==========

subreddit_dict = Counter(subreddits)
print("各板块样本数（前15个）：", subreddit_dict)
print("总样本数：", len(new_samples))

kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
i = 1
for not_test_index, test_index in kf.split(X=new_samples, y=subreddits, groups=subreddits):
    # 注意：原代码此处是iloc_list(samples, ...)，需改为iloc_list(new_samples, ...)（原代码bug修复）
    not_test_samples, test_samples = iloc_list(new_samples, not_test_index), iloc_list(new_samples, test_index)

    not_test_labels = iloc_list(subreddits, not_test_index)
    print(f"第{i}折 - 非测试集板块分布：", Counter(not_test_labels))

    train_samples, dev_samples, train_labels, dev_labels = train_test_split(
        not_test_samples, not_test_labels,
        test_size=0.125, random_state=24,
        shuffle=True, stratify=not_test_labels
    )

    print(f"第{i}折 - Train {len(train_samples)} Dev {len(dev_samples)} Test {len(test_samples)}")
    if not os.path.exists("../data/reddit/split/"):
        os.makedirs("../data/reddit/split/")
    split_dir = os.path.join("../data/reddit/split/", str(i))
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    # 写入划分后的数据集
    with open(os.path.join(split_dir, "train_ids.json"), "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=4)  # 原代码误写为not_test_samples，此处修复
    with open(os.path.join(split_dir, "dev_ids.json"), "w", encoding="utf-8") as f:
        json.dump(dev_samples, f, indent=4)  # 原代码误写为test_samples，此处修复
    with open(os.path.join(split_dir, "test_ids.json"), "w", encoding="utf-8") as f:
        json.dump(test_samples, f, indent=4)

    i += 1