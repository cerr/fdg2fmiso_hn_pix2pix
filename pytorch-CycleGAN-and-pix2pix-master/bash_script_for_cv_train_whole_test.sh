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

# python train_test_cross_validation_FDG_tbr_whole.py \
# --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_whole_70_10_20 \
# --name fdg_tbr_pix2pix \
# --model pix2pix_32_fdg_tbr_three_down_layer_add_block_number_pixel_d \
# --dataset_mode fdg_tbr \
# --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdg_TBR_pix2pix_l1_l2_tumor_pixel_D_three_down_layer_whole \
# --batch_size 1 



python show_test_results_FDG_tbr_no_augmentation.py \
--dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_whole_70_10_20 \
--name fdg_tbr_pix2pix \
--model pix2pix_32_fdg_tbr_three_down_layer_add_block_number_pixel_d \
--dataset_mode fdg_tbr \
--checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdg_TBR_pix2pix_l1_l2_tumor_pixel_D_three_down_layer_whole


# # this is for FDGk1 to TBR model training whole data


# python train_test_cross_validation_FDGk1_k3_whole.py \
# --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_whole_70_10_20 \
# --name fdgk1_tbr_pix2pix \
# --model pix2pix_32_tbr_three_down_layer_add_block_number_pixel_d \
# --dataset_mode fdgk1_fmiso \
# --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_TBR_pix2pix_l1_l2_tumor_pixel_D_three_down_layer_whole \
# --batch_size 1 



# python show_test_results_FDGk1_k3_no_agumentation.py \
# --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_whole_70_10_20 \
# --name fdgk1_tbr_pix2pix \
# --model pix2pix_32_tbr_three_down_layer_add_block_number_pixel_d \
# --dataset_mode fdgk1_fmiso \
# --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_TBR_pix2pix_l1_l2_tumor_pixel_D_three_down_layer_whole


# for ((i=1;i<=5;i++))
# do 
#     python train_test_cross_validation_FDGk1_k3.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_whole_70_10_20/cv_5Fold/"$i" \
#     --name fdgk1_tbr_pix2pix \
#     --model pix2pix_32_tbr_three_down_layer_add_block_number_pixel_d \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_TBR_pix2pix_fold"$i"_l1_l2_tumor_pixel_D_three_down_layer_whole_cv \
#     --batch_size 1 
# done

# for ((i=1;i<=5;i++))
# do 
#     python show_test_results_FDGk1_k3_no_agumentation_cv_test.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_whole_70_10_20 \
#     --phase cv_test \
#     --name fdgk1_tbr_pix2pix \
#     --model pix2pix_32_tbr_three_down_layer_add_block_number_pixel_d \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_TBR_pix2pix_fold"$i"_l1_l2_tumor_pixel_D_three_down_layer_whole_cv 
# done

# for ((i=1;i<=5;i++))
# do 
#     python show_test_results_FDGk1_k3_no_agumentation.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_whole_70_10_20/cv_5Fold/"$i" \
#     --name fdgk1_tbr_pix2pix \
#     --model pix2pix_32_tbr_three_down_layer_add_block_number_pixel_d \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_TBR_pix2pix_fold"$i"_l1_l2_tumor_pixel_D_three_down_layer_whole_cv 
# done

# for ((i=1;i<=5;i++))
# do 
#     python train_test_cross_validation_FDGk1_k3.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_5Fold/"$i" \
#     --name fdgk1_tbr_pix2pix \
#     --model pix2pix_32_tbr_three_down_layer_four_by_four \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_TBR_pix2pix_fold"$i"_l1_l2_tumor_4_three_down_layer \
#     --batch_size 1
# done

# for ((i=1;i<=5;i++))
# do 
#     python show_test_results_FDGk1_k3.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_5Fold/"$i" \
#     --name fdgk1_tbr_pix2pix \
#     --model pix2pix_32_tbr_three_down_layer_four_by_four \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_TBR_pix2pix_fold"$i"_l1_l2_tumor_4_three_down_layer 
# done

# for ((i=1;i<=5;i++))
# do 
#     python train_test_cross_validation_FDGk1_k3.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_5Fold/"$i" \
#     --name fdgk1_tbr_pix2pix \
#     --model pix2pix_32_tbr_wgan_three_down_layer \
#     --dataset_mode fdgk1_fmiso \
#     --lambda_L1 1000 \
#     --lambda_L2 500 \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_TBR_pix2pix_fold"$i"_large_l1_l2_tumor_16_wgan_three_down_layer \
#     --batch_size 1
# done

# for ((i=1;i<=5;i++))
# do 
#     python show_test_results_FDGk1_k3.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_5Fold/"$i" \
#     --name fdgk1_tbr_pix2pix \
#     --model pix2pix_32_tbr_wgan_three_down_layer \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_TBR_pix2pix_fold"$i"_large_l1_l2_tumor_16_wgan_three_down_layer 
# done
# #five down layer, 1by1 in bottleneck in unet
# for ((i=1;i<=5;i++))
# do 
#     python train_test_cross_validation_FDGk1_k3.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_5Fold/"$i" \
#     --name fdgk1_tbr_pix2pix \
#     --model pix2pix_32_tbr \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_TBR_pix2pix_fold"$i"_l1_l2_tumor_16 \
#     --batch_size 1
# done

# for ((i=1;i<=5;i++))
# do 
#     python show_test_results_FDGk1_k3.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_5Fold/"$i" \
#     --name fdgk1_tbr_pix2pix \
#     --model pix2pix_32_tbr \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_TBR_pix2pix_fold"$i"_l1_l2_tumor_16 
# done
# #five down layer, 1by1 at bottleneck in unet, add se in all layer
# for ((i=1;i<=5;i++))
# do 
#     python train_test_cross_validation_FDGk1_k3.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_5Fold/"$i" \
#     --name fdgk1_tbr_pix2pix \
#     --model pix2pix_32_tbr_se_all_layer \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_TBR_pix2pix_fold"$i"_l1_l2_tumor_16_SE_all_layer \
#     --batch_size 1
# done

# for ((i=1;i<=5;i++))
# do 
#     python show_test_results_FDGk1_k3.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_5Fold/"$i" \
#     --name fdgk1_tbr_pix2pix \
#     --model pix2pix_32_tbr_se_all_layer \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_TBR_pix2pix_fold"$i"_l1_l2_tumor_16_SE_all_layer 
# done

# this is for FDGk1 to k3 model training where k3 is NOT log transformed transformed
# for ((i=1;i<=5;i++))
# do 
#     python train_test_cross_validation_FDGk1_k3.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_k3_patch_pair_5Fold/"$i" \
#     --name fdgk1_k3_pix2pix \
#     --model pix2pix_32_fdg_k1_k3 \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_k3_pix2pix_fold"$i"_l1_l2_tumor_SE_16 \
#     --batch_size 1
# done

# for ((i=1;i<=5;i++)); do 
#     python train_test_cross_validation.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDG_k1_patch_pair_no_margin_5Fold/"$i" \
#     --name fdg_k1_pix2pix \
#     --model pix2pix_32 \
#     --dataset_mode fdg_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdg_k1_pix2pix_no_margin_fold"$i"_l1_l2_tumor_SE_16 \
#     --batch_size 1
# done