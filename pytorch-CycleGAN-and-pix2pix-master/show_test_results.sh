#bsub -M 30GB -q gpuqueue -n 1 -W 168:00 -gpu "num=1" -R V100 -Is /bin/bash
#bsub -M 20GB -n 1 -gpu "num=1" -q gpuqueue -m lx-gpu -W 1:00  -Is /bin/bash

source /home/zhaow2/anaconda3/bin/activate 
conda activate tf1
cd /home/zhaow2/pytorch-CycleGAN-and-pix2pix-master

for ((i=1;i<=1;i++))
do 
    python show_test_results_FDGk1_k3.py \
    --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_k3_patch_pair_5Fold/"$i" \
    --name fdgk1_k3_pix2pix \
    --model pix2pix_32_fdg_k1_k3_ignore_test_k3 \
    --dataset_mode fdgk1_fmiso \
    --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_k3_pix2pix_ignore_00025_k3_fold"$i"_l1_l2_tumor_SE_16 
done

# for ((i=1;i<=5;i++))
# do 
#     python show_test_results_FDGk1_k3.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_k3_patch_pair_5Fold/"$i" \
#     --name fdgk1_k3_pix2pix \
#     --model pix2pix_32_fdg_k1_k3_ignore_zero_k3 \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_k3_pix2pix_ignore_zero_k3_fold"$i"_l1_l2_tumor_SE_16 
# done

# for ((i=1;i<=5;i++))
# do 
#     python show_test_results_FDGk1_k3.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_k3_patch_pair_k3_log_transformed_5Fold/"$i" \
#     --name fdgk1_k3_pix2pix \
#     --model pix2pix_32_fdg_k1_k3 \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_k3_pix2pix_k3_log_transformed_fold"$i"_l1_l2_tumor_SE_16 
# done

# for ((i=1;i<=5;i++))
# do 
#     python show_test_results_FDGk1_k3_ignore_zero_k3.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_k3_patch_pair_k3_log_transformed_5Fold/"$i" \
#     --name fdgk1_k3_pix2pix \
#     --model pix2pix_32_fdg_k1_k3_ignore_zero_k3 \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_k3_pix2pix_k3_log_transformed_ignore_zero_k3_fold"$i"_l1_l2_tumor_SE_16_fixed 
# done
# for i in 3
# do 
#     python show_test_results.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_k3_patch_pair_no_margin_5Fold/"$i" \
#     --name fdgk1_k3_pix2pix \
#     --model pix2pix_32 \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_k3_pix2pix_no_margin_fold"$i"_l1_l2_tumor_SE_16 
# done

# for i in 1 3
# do 
#     python show_test_results_k3_log_transformed.py \
#     --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_k3_patch_pair_k3_log_transformed_5Fold/"$i" \
#     --name fdgk1_k3_pix2pix \
#     --model pix2pix_32 \
#     --dataset_mode fdgk1_fmiso \
#     --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdgk1_k3_pix2pix_k3_log_transformed_fold"$i"_l1_l2_tumor_SE_16 
# done