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
conda activate base
# conda activate tf1
cd /home/zhaow2/pytorch-CycleGAN-and-pix2pix-master



# # this is for FDGk1 to TBR model training whole data


python train_test_cross_validation_FDG_tbr_whole_weighted_l1.py \
--dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_whole_70_10_20 \
--name fdg_tbr_pix2pix \
--model pix2pix_32_fdg_tbr_three_down_layer_add_block_number_pixel_d_weighted_l1 \
--dataset_mode fdg_tbr \
--checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdg_TBR_pix2pix_l1_l2_tumor_pixel_D_three_down_layer_whole_weighted_l1 \
--batch_size 1 

python show_test_results_FDG_tbr_no_augmentation_weighted_l1.py \
--dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_whole_70_10_20 \
--name fdg_tbr_pix2pix \
--model pix2pix_32_fdg_tbr_three_down_layer_add_block_number_pixel_d_weighted_l1 \
--dataset_mode fdg_tbr \
--checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdg_TBR_pix2pix_l1_l2_tumor_pixel_D_three_down_layer_whole_weighted_l1


python train_test_cross_validation_FDG_tbr_whole_weighted_l1.py \
--dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_whole_70_10_20 \
--name fdg_tbr_pix2pix \
--model pix2pix_32_fdg_tbr_three_down_layer_four_by_four_weighted_l1 \
--dataset_mode fdg_tbr \
--checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdg_TBR_pix2pix_l1_l2_tumor_four_by_four_three_down_layer_whole_weighted_l1 \
--batch_size 1 


python show_test_results_FDG_tbr_no_augmentation_weighted_l1.py \
--dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_whole_70_10_20 \
--name fdg_tbr_pix2pix \
--model pix2pix_32_fdg_tbr_three_down_layer_four_by_four_weighted_l1 \
--dataset_mode fdg_tbr \
--checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdg_TBR_pix2pix_l1_l2_tumor_four_by_four_three_down_layer_whole_weighted_l1

python train_test_cross_validation_FDG_tbr_whole_weighted_l1.py \
--dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_whole_70_10_20 \
--name fdg_tbr_pix2pix \
--model pix2pix_32_fdg_tbr_weighted_l1 \
--dataset_mode fdg_tbr \
--checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdg_TBR_pix2pix_l1_l2_tumor_whole_weighted_l1 \
--batch_size 1 

python show_test_results_FDG_tbr_no_augmentation_weighted_l1.py \
--dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_whole_70_10_20 \
--name fdg_tbr_pix2pix \
--model pix2pix_32_fdg_tbr_weighted_l1 \
--dataset_mode fdg_tbr \
--checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdg_TBR_pix2pix_l1_l2_tumor_whole_weighted_l1

for ((i=1;i<=5;i++))
do 
    python train_test_cross_validation_FDG_tbr_weighted_l1.py \
    --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_whole_70_10_20/cv_5Fold/"$i" \
    --name fdg_tbr_pix2pix \
    --model pix2pix_32_fdg_tbr_three_down_layer_add_block_number_pixel_d_weighted_l1 \
    --dataset_mode fdg_tbr \
    --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdg_TBR_pix2pix_fold"$i"_l1_l2_tumor_pixel_D_three_down_layer_whole_cv_weighted_l1 \
    --batch_size 1 
done

for ((i=1;i<=5;i++))
do 
    python show_test_results_FDG_tbr_no_augmentation_weighted_l1.py \
    --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_whole_70_10_20 \
    --phase cv_test \
    --name fdg_tbr_pix2pix \
    --model pix2pix_32_fdg_tbr_three_down_layer_add_block_number_pixel_d_weighted_l1 \
    --dataset_mode fdg_tbr \
    --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdg_TBR_pix2pix_fold"$i"_l1_l2_tumor_pixel_D_three_down_layer_whole_cv_weighted_l1 
done