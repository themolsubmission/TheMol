#data_path=/home/csy/work1/3D/Uni-Mol/unimol/example_data/ligands
data_path=/home/csy/work1/3D/TheMol/GEOM_dataset
save_dir=./saveGEOM # replace to your save path
arch=unimol_Optimal_padding2
logfile=${save_dir}/${arch}.log
n_gpu=2
MASTER_PORT=$RANDOM
lr=1e-4
wd=1e-4
batch_size=512
update_freq=1
masked_token_loss=1
masked_coord_loss=10
masked_dist_loss=10
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
mask_prob=0.15
only_polar=0
noise_type="uniform"
noise=1.0
seed=1
warmup_steps=10000
max_steps=1000000

mkdir -p ${save_dir}

cp $0 ${save_dir}

export TORCH_AUTOGRAD_DETECT_ANOMALY=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="2,3"
export TORCH_CUDA_ARCH_LIST="8.9"
export PYTHONUNBUFFERED=1

python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --find-unused-parameters --task unimol --task-sub generative --training flow_only --loss unimol_MAE2 --arch ${arch} --wandb disabled \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
       --update-freq $update_freq --seed $seed \
       --tensorboard-logdir $save_dir/tsb \
       --max-update $max_steps --log-interval 100 --log-format simple \
       --save-interval-updates 5000 --validate-interval-updates 1000 --keep-interval-updates 10 --no-epoch-checkpoints  \
       --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
       --decoder-x-norm-loss $x_norm_loss --decoder-delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
       --encoder-x-norm-loss $x_norm_loss --encoder-delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
       --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
       --encoder-unmasked-tokens-only \
       --encoder-embed-dim 128 --encoder-attention-heads 8 --encoder-ffn-embed-dim 128 \
       --encoder-layers 10 --decoder-layers 5 --decoder-ffn-embed-dim 128 --decoder-attention-heads 8 \
       --save-dir $save_dir  --only-polar $only_polar > ${logfile} 2>&1


# saveGEOM >> Dual
# saveGEOM_ >> Flow_only