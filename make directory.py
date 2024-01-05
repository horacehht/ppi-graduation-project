import os
import os.path as osp

base_path = './result_save'

seeds = [7, 17, 27, 37, 47, 57, 67, 87, 3407]
for seed in seeds:
    save_path = osp.join(base_path, "pipr_training_seed_{}".format(str(seed)))
    os.mkdir(save_path)