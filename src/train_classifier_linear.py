import config
from framework import Framework
import argparse
import torch
import os

from utils.utils import seed_everything
import warnings

warnings.filterwarnings("ignore")
import transformers

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# transformers.logging.set_verbosity_error()
parser = argparse.ArgumentParser()
# ========== 关键修改1：device默认值从1改成0（适配AutoDL单GPU） ==========
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--seed', type=int, default=1024)
parser.add_argument('--model_path', type=str, default='../models')
parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='name of the model')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--bert_lr', type=float, default=2e-5)
parser.add_argument('--warmup_proportion', type=float, default=0.1)
parser.add_argument('--max_norm', type=float, default=1.0)
parser.add_argument('--init_mode', type=str, default='xavier_normal')
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_epoch', type=int, default=5)
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--train_prefix', type=str, default='train')
parser.add_argument('--dev_prefix', type=str, default='dev')
parser.add_argument('--test_prefix', type=str, default='test')
parser.add_argument('--max_len', type=int, default=384)
parser.add_argument('--clause_max_len', type=int, default=64)
parser.add_argument('--attention_head', type=int, default=12)
parser.add_argument('--period', type=int, default=50)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--data_path', type=str, default='../data/reddit/split')
parser.add_argument('--log_path', type=str, default='log')
parser.add_argument('--result_path', type=str, default='./result/')
args = parser.parse_args()

if __name__ == '__main__':
    con = config.Config(args)

    # ========== 关键修改2：增加GPU设备有效性检查（避免无效设备报错） ==========
    if torch.cuda.is_available():
        # 获取可用的GPU数量，确保device序号有效
        available_gpus = torch.cuda.device_count()
        if con.device >= available_gpus:
            # 如果指定的device无效，自动切换到第0号GPU
            con.device = 0
            print(f"警告：指定的GPU设备 {args.device} 不存在，自动切换到GPU 0")
        # 设置有效GPU + 限制显存使用比例
        torch.cuda.set_device(con.device)
        torch.cuda.set_per_process_memory_fraction(0.8, device=con.device)
    else:
        print("警告：未检测到CUDA GPU，将使用CPU训练（速度极慢）")

    print(con.root)
    seed_everything(con.seed)
    con.log_dir = os.path.join(args.log_path, str(con.bert_lr) + str(con.lr))
    con.checkpoint_dir = os.path.join(con.log_dir, "checkpoint")

    # 初始化前清空显存
    torch.cuda.empty_cache()
    fw = Framework(con)
    torch.cuda.empty_cache()
    fw.train()