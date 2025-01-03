#bsub -M 30GB -q gpuqueue -n 1 -W 168:00 -gpu "num=1" -R V100 -Is /bin/bash
#bsub -M 20GB -n 1 -gpu "num=1" -q gpuqueue -m lx-gpu -W 1:00  -Is /bin/bash
#ps -ef |grep nohup to get the id of background nohup task id
#kill -9 process_id to kill

# !/bin/bash

# BSUB -J wgan_up_1

# BSUB -n 1

# BSUB -M 64G

# BSUB -W 150:00

# BSUB -q gpuqueue

# BSUB -gpu "num=1"

# BSUB -R V100

# BSUB -o %J.stdout

# BSUB -eo %J.stderr
source /home/zhaow2/anaconda3/bin/activate 
conda activate tf1
cd /home/zhaow2/pytorch-CycleGAN-and-pix2pix-master

# this is for FDGk1 to k3 model training where k3 is log transformed
# for ((i=1;i<=5;i++))
# do 
#     python train_test_cross_validation_FDGk1_k3.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_k3_patch_pair_k3_log_transformed_5Fold/"$i" \
#     --name fdgk1_k3_pix2pix \
#     --model pix2pix_32_fdg_k1_k3 \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_k3_pix2pix_k3_log_transformed_fold"$i"_l1_l2_tumor_SE_16 \
#     --batch_size 1
# done

# this is for FDGk1 to k3 model training where k3 is NOT log transformed transformed
for ((i=1;i<=5;i++))
do 
    python train_test_cross_validation_FDGk1_k3.py \
    --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_k3_patch_pair_5Fold/"$i" \
    --name fdgk1_k3_pix2pix \
    --model pix2pix_32_fdg_k1_k3 \
    --dataset_mode fdgk1_fmiso \
    --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_k3_pix2pix_fold"$i"_l1_l2_tumor_SE_16 \
    --batch_size 1
done

# for ((i=1;i<=5;i++)); do 
#     python train_test_cross_validation.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDG_k1_patch_pair_no_margin_5Fold/"$i" \
#     --name fdg_k1_pix2pix \
#     --model pix2pix_32 \
#     --dataset_mode fdg_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdg_k1_pix2pix_no_margin_fold"$i"_l1_l2_tumor_SE_16 \
#     --batch_size 1
# done