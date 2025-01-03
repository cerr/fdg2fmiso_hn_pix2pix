#bsub -M 30GB -q gpuqueue -n 1 -W 168:00 -gpu "num=1" -R V100 -Is /bin/bash
#nohup long-running-command & keep the process running after close the terminal
source /home/zhaow2/anaconda3/bin/activate 
conda activate tf1
cd /home/zhaow2/pytorch-CycleGAN-and-pix2pix-master

for ((i=1;i<=5;i++)); do 
    python test.py \
    --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDG_k1_patch_pair_5Fold/"$i"/train \
    --name fdg_fmisok1_pix2pix \
    --model pix2pix_32 \
    --dataset_mode fdg_fmiso \
    --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdg_fmisok1_pix2pix_fold"$i" \
    --batch_size 2
done
# python train.py \
# --dataroot /lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDG_k1_patch_pair_5Fold/1/train \
# --name fdg_fmisok1_pix2pix \
# --model pix2pix_32 \
# --dataset_mode fdg_fmiso \
# --checkpoints_dir /lila/home/zhaow2/pytorch-CycleGAN-and-pix2pix-master/checkpoints/fdg_fmisok1_pix2pix_fold1 \
# --batch_size 2 \
#--gpu_ids 1 \
# --niter 20 \
# --loadSize 192 \
# --fineSize 192 \
# --loss dice \
#--model_type standard \
#--fold_file /lila/data/deasy/nazib/trainvalset-Lev-fold1.csv \
#--gan_train True \
#--gan_loss wgan_up \
#--lr 0.00005