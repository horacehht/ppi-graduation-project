import subprocess
import os.path as osp

seeds = [7, 17, 27, 37, 47, 57, 67, 87, 3407]  # PR的seed
ood_seeds = [2, 3, 4, 5, 6]  # OOD的seed
robust_seeds = [11, 12, 13, 14, 15]  # robust的seed

ppi_path = "./protein_info/protein.actions.SHS27k.STRING.pro2.txt"
pseq_path = "./protein_info/protein.SHS27k.sequences.dictionary.pro3.tsv"
vec_path = "./protein_info/vec5_CTC.txt"


# PR曲线图
save_path = "./result_save"
index_folder = "./train_val_split_data"
for seed in seeds:
    index_path = osp.join(index_folder, "train_val_split_{}.json".format(str(seed)))
    # 执行训练脚本
    train_command = f"python gnn_train.py \
    --ppi_path {ppi_path} \
    --pseq_path {pseq_path} \
    --vec_path {vec_path} \
    --index_path {index_path} \
    --save_path {save_path} \
    --seed {seed} \
    " # 每一行参数后面加\也可以
    print("-"*25, "TRAINING", "-"*25)
    print(f"Running training with seed {seed}")
    subprocess.run(train_command, shell=True)
    
    # 执行测试脚本
    test_command = f"""python gnn_test.py \
    --ppi_path {ppi_path} \
    --pseq_path {pseq_path} \
    --vec_path {vec_path} \
    --save_path {save_path} \
    --index_path {index_path} \
    --seed {seed}
    """
    print("-"*25, "TESTING", "-"*25)
    print(f"Running testing with seed {seed}")
    subprocess.run(test_command, shell=True)


# OOD性能
cases = ['bfs-0.3', 'bfs-0.4', 'dfs-0.3', 'dfs-0.4', 'random-0.35']
# 一个case下跑多次种子进行验证
for case in cases:
    for seed in ood_seeds:
        split, test_ratio = case.split('-')
        test_ratio = float(test_ratio)
        save_path = f"./OOD_result_save_{case}"  # 每一个case有一个save path
        index_folder = f"./split_{case}"
        index_path = osp.join(index_folder, "train_val_split_{}.json".format(str(seed)))
        
        # 执行训练脚本
        train_command = f"python gnn_train.py \
        --ppi_path {ppi_path} \
        --pseq_path {pseq_path} \
        --vec_path {vec_path} \
        --index_path {index_path} \
        --save_path {save_path} \
        --seed {seed} \
        " # 每一行参数后面加\也可以
        print("-"*25, "TRAINING", "-"*25)
        print(f"Running training with seed {seed}")
        subprocess.run(train_command, shell=True)
        
        
        # 执行测试脚本
        test_command = f"""python gnn_test.py \
        --ppi_path {ppi_path} \
        --pseq_path {pseq_path} \
        --vec_path {vec_path} \
        --save_path {save_path} \
        --index_path {index_path} \
        --seed {seed}
        """  
        print("-"*25, "TESTING", "-"*25)
        print(f"Running testing with seed {seed}")
        subprocess.run(test_command, shell=True)


# robustness
perturb_ratios = [0.2, 0.4, 0.6]
for perturb_ratio in perturb_ratios:
    for seed in robust_seeds:
        save_path = f"./robust_result_save_perturb-{perturb_ratio}"  # 每一个扰动比例一个save_path
        index_folder = f"./perturb_{perturb_ratio}"
        index_path = osp.join(index_folder, "train_val_split_{}.json".format(str(seed)))
        # 执行训练脚本
        train_command = f"python gnn_train.py \
        --ppi_path {ppi_path} \
        --pseq_path {pseq_path} \
        --vec_path {vec_path} \
        --index_path {index_path} \
        --save_path {save_path} \
        --seed {seed} \
        " # 每一行参数后面加\也可以
        print("-"*25, "TRAINING", "-"*25)
        print(f"Running training with seed {seed}")
        subprocess.run(train_command, shell=True)
        
        # 执行测试脚本
        test_command = f"""python gnn_test.py \
        --ppi_path {ppi_path} \
        --pseq_path {pseq_path} \
        --vec_path {vec_path} \
        --save_path {save_path} \
        --index_path {index_path} \
        --seed {seed}
        """
        print("-"*25, "TESTING", "-"*25)
        print(f"Running testing with seed {seed}")
        subprocess.run(test_command, shell=True)