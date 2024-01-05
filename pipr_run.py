import subprocess
import os.path as osp


# seeds = [7, 17, 27, 37, 47, 57, 67, 87, 3407]  # PR曲线 seed
ood_seeds = [2, 3, 4, 5, 6] #  DFS-0.4是最简单的case，此处跟high_run保持一致
robust_seeds = [11, 12, 13, 14, 15]

ppi_path = "./protein_info/protein.actions.SHS27k.STRING.pro2.txt"
pseq_path = "./protein_info/protein.SHS27k.sequences.dictionary.pro3.tsv"
vec_path = "./protein_info/vec5_CTC.txt"
save_path = "./result_save"
index_folder = "./train_val_split_data"


# PR曲线图，跑9次模型训练
for seed in seeds:
    index_path = osp.join(index_folder, "train_val_split_{}.json".format(str(seed)))
    train_command = f"""python pipr_train.py \
    --ppi_path {ppi_path} \
    --pseq_path {pseq_path} \
    --vec_path {vec_path} \
    --index_path {index_path} \
    --save_path {save_path} \
    --seed {seed} \
    """  # 要写成一行，否则不能正常读入参数
    print("-"*25, "TRAINING", "-"*25)
    print(f"Running training with seed {seed}")
    subprocess.run(train_command, shell=True)
    
    test_command = f"""python pipr_test.py \
    --ppi_path {ppi_path} \
    --pseq_path {pseq_path} \ 
    --vec_path {vec_path} \
    --index_path {index_path} \ 
    --save_path {save_path} \
    --seed {seed} \
    """
    print("-"*25, "TESTING", "-"*25)
    print(f"Running testing with seed {seed}")
    subprocess.run(test_command, shell=True)

# ---------------PIPR按HIGH-PPI论文里说只用跑DFS-0.4这个实验，性能最好----------------
index_folder = "./split_dfs-0.4"
save_path = "./OOD_result_save_dfs-0.4"
for seed in ood_seeds:
    index_path = osp.join(index_folder, "train_val_split_{}.json".format(str(seed)))
    train_command = f"""python pipr_train.py \
        --ppi_path {ppi_path} \
        --pseq_path {pseq_path} \
        --vec_path {vec_path} \
        --index_path {index_path} \
        --save_path {save_path} \
        --seed {seed}  \
    """
    print("-"*25, "TRAINING", "-"*25)
    print(f"Running training with seed {seed}")
    subprocess.run(train_command, shell=True)

    test_command = f"""python pipr_test.py \
        --ppi_path {ppi_path} \
        --pseq_path {pseq_path} \
        --vec_path {vec_path} \
        --index_path {index_path} \
        --save_path {save_path} \
        --seed {seed}  \
    """
    print("-"*25, "TESTING", "-"*25)
    print(f"Running testing with seed {seed}")
    subprocess.run(test_command, shell=True)


# robustness

perturb_ratios = [0.2, 0.4, 0.6]
for perturb_ratio in perturb_ratios:
    for seed in robust_seeds:
        index_folder = f"./perturb_{perturb_ratio}"
        save_path = f"./robust_result_save_perturb-{perturb_ratio}"
        index_path = osp.join(index_folder, "train_val_split_{}.json".format(str(seed)))
        train_command = f"""python pipr_train.py \
            --ppi_path {ppi_path} \
            --pseq_path {pseq_path} \
            --vec_path {vec_path} \
            --index_path {index_path} \
            --save_path {save_path} \
            --seed {seed}  \
        """
        print("-"*25, "TRAINING", "-"*25)
        print(f"Running training with seed {seed}")
        subprocess.run(train_command, shell=True)

        test_command = f"""python pipr_test.py \
            --ppi_path {ppi_path} \
            --pseq_path {pseq_path} \
            --vec_path {vec_path} \
            --index_path {index_path} \
            --save_path {save_path} \
            --seed {seed}  \
        """
        print("-"*25, "TESTING", "-"*25)
        print(f"Running testing with seed {seed}")
        subprocess.run(test_command, shell=True)