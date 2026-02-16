import numpy as np
import os  # 新增：用于路径处理和文件检查

def read_txt(file_name):
    if not os.path.exists(file_name):
        print(f"警告：日志文件不存在 {file_name}")
        return []  # 不存在时返回空列表
    with open(file_name, 'r', encoding='utf-8') as f:  # 新增编码参数
        lines = f.readlines()
        return [line.strip() for line in lines]

lines = []
for i in range(1, 6):
    # 用os.path.join处理路径，适配Windows
    log_dir = os.path.join('./log', str(i), '2e-050.0001')
    file_name = 'LOG_bert-base-uncased_BERTLR_2.000000e-05_LR_1.000000e-04_BS_5'
    file_path = os.path.join(log_dir, file_name)
    line = read_txt(file_path)
    if line:  # 仅添加非空内容
        lines.extend(line[-9:])

# 新增：检查是否有有效日志
if not lines:
    print("错误：未找到有效日志文件，请先确保训练成功")
    exit(1)

precisions_subreddit = {'all':[], 'android':[], 'apple':[], 'technology':[], 'dota2':[], 'playstation':[], 'movies':[], 'nba':[]}
recalls_subreddit = {'all':[], 'android':[], 'apple':[], 'technology':[], 'dota2':[], 'playstation':[], 'movies':[], 'nba':[]}
f1s_subreddit = {'all':[], 'android':[], 'apple':[], 'technology':[], 'dota2':[], 'playstation':[], 'movies':[], 'nba':[]}

for line in lines:
    line_parts = line.split(',')
    # 新增：检查日志格式是否正确
    if len(line_parts) < 4:
        print(f"警告：日志格式错误，跳过此行：{line}")
        continue
    subreddit = line_parts[1].strip()
    # 解析精度、召回率、F1（根据实际日志格式调整索引）
    try:
        precision = float(line_parts[2].split(':')[1].strip())
        recall = float(line_parts[3].split(':')[1].strip())
        f1 = float(line_parts[4].split(':')[1].strip())
    except (IndexError, ValueError) as e:
        print(f"警告：解析日志失败 {e}，跳过此行：{line}")
        continue
    # 分类存储
    if 'doc_acc' in subreddit:
        precisions_subreddit['all'].append(precision)
        recalls_subreddit['all'].append(recall)
        f1s_subreddit['all'].append(f1)
    else:
        subreddit_name = subreddit.split(' ')[1].strip()
        if subreddit_name in precisions_subreddit:
            precisions_subreddit[subreddit_name].append(precision)
            recalls_subreddit[subreddit_name].append(recall)
            f1s_subreddit[subreddit_name].append(f1)

# 生成输出
output_lines = []
for subreddit in precisions_subreddit:
    if precisions_subreddit[subreddit]:  # 仅处理有数据的条目
        avg_p = np.mean(precisions_subreddit[subreddit])
        avg_r = np.mean(recalls_subreddit[subreddit])
        avg_f1 = np.mean(f1s_subreddit[subreddit])
        output_lines.append(f'{subreddit},{avg_p:.4f},{avg_r:.4f},{avg_f1:.4f}')

with open('./log/result.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))