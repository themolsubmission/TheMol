#!/bin/bash

# ==============================================================================
# Maximum Parallel Grid Search Runner
# - Runs 3-4 experiments simultaneously per GPU
# - Optimized for 46GB GPU memory
# - Auto-balances small and large datasets to avoid OOM
# ==============================================================================

# Activate conda environment (modify for your setup)
# source /path/to/anaconda3/etc/profile.d/conda.sh
# conda activate your_env

# Change to project directory
cd "$(dirname "$0")/.."

# Common configuration - MODIFY THESE PATHS
data_path="./datasets/moleculenet"  # Path to MoleculeNet dataset
dict_name="dict.txt"
weight_path="./checkpoints/checkpoint_last.pt"  # Path to pretrained checkpoint
n_gpu=1
arch=unimol_Optimal_padding_Dual
epoch=200
conf_size=1
seed=1553
delta_pair_repr_norm_loss=0.01

# Environment variables
export TORCH_AUTOGRAD_DETECT_ANOMALY=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export TORCH_CUDA_ARCH_LIST="8.9"

# Hyperparameter grid
learning_rates=(5e-5 8e-5 1e-4)
batch_sizes=(32 64)
pooler_dropouts=(0.0 0.1 0.2 0.5)
warmup_ratios=(0.0 0.06 0.1)

# Dataset configurations with memory estimates
# Format: name|task_num|loss_func|metric|base_port|mem_category
# mem_category: small (2-12 classes), medium (17-27 classes), large (128-617 classes)
declare -a datasets=(
    "bace|2|finetune_cross_entropy|valid_agg_auc|10400|small"
    "clintox|2|multi_task_BCE|valid_agg_auc|10700|small"
    "hiv|2|finetune_cross_entropy|valid_agg_auc|12000|small"
    "lipo|1|finetune_smooth_mae|valid_agg_rmse|15000|small"
    "tox21|12|multi_task_BCE|valid_agg_auc|11000|small"
    "muv|17|multi_task_BCE|valid_agg_auc|13000|medium"
    "sider|27|multi_task_BCE|valid_agg_auc|10086|medium"
    "pcba|128|multi_task_BCE|valid_agg_auc|12500|large"
    "toxcast|617|multi_task_BCE|valid_agg_auc|11500|large"
)

# Maximum concurrent jobs per GPU based on dataset size
# small tasks: up to 4 concurrent (bs=32)
# medium tasks: up to 3 concurrent
# large tasks: up to 2 concurrent
MAX_CONCURRENT_SMALL=4
MAX_CONCURRENT_MEDIUM=3
MAX_CONCURRENT_LARGE=2

# Function to check if experiment is already completed
# Changed: Now checks for checkpoint_best.pt instead of log file
# is_experiment_completed() {
#     local save_dir=$1
#     local checkpoint_file="${save_dir}/checkpoint_best.pt"
#     if [ -f "$checkpoint_file" ]; then
#         return 0  # Completed - best checkpoint exists
#     fi
#     return 1  # Not completed
# }
is_experiment_completed() {
    local save_dir=$1
    local checkpoint_file="${save_dir}/checkpoint_best.pt"
    local logfile="${save_dir}/${arch}.log"

    # Completion check: 1) checkpoint_best.pt exists AND 2) log contains "done training in"
    if [ -f "$checkpoint_file" ] && [ -f "$logfile" ] && grep -q "done training in" "$logfile"; then
        return 0  # Completed
    fi

    return 1  # Not completed -> will be retrained
}


# Function to run a single experiment
run_experiment() {
    local gpu_id=$1
    local task_name=$2
    local task_num=$3
    local loss_func=$4
    local metric=$5
    local base_port=$6
    local lr=$7
    local batch_size=$8
    local dropout=$9
    local warmup=${10}
    local exp_id=${11}

    local base_save_dir="./save_${task_name}"
    local save_dir="${base_save_dir}/lr${lr}_bs${batch_size}_drop${dropout}_warm${warmup}"
    local logfile="${save_dir}/${arch}.log"

    # Check if already completed
    if is_experiment_completed "$save_dir"; then
        return 0
    fi

    mkdir -p "${save_dir}"

    # Calculate unique port
    local MASTER_PORT
    if [ $gpu_id -eq 2 ]; then
        MASTER_PORT=$((base_port + exp_id))
    else
        MASTER_PORT=$((base_port + 1000 + exp_id))
    fi

    echo "[START] GPU${gpu_id} | ${task_name}: lr=${lr}, bs=${batch_size}, drop=${dropout}, warm=${warmup} | Port: ${MASTER_PORT}" | tee -a "gpu${gpu_id}_worker.log"

    CUDA_VISIBLE_DEVICES=$gpu_id python -m torch.distributed.launch \
        --nproc_per_node=$n_gpu --master_port=$MASTER_PORT \
        $(which unicore-train) $data_path \
        --task-name $task_name \
        --user-dir ./unimol \
        --train-subset train \
        --valid-subset valid \
        --conf-size $conf_size \
        --num-workers 8 \
        --ddp-backend=c10d \
        --dict-name $dict_name \
        --find-unused-parameters \
        --task mol_finetune \
        --loss $loss_func \
        --arch ${arch} \
        --classification-head-name $task_name \
        --num-classes $task_num \
        --decoder-delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
        --encoder-delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
        --optimizer adam \
        --adam-betas "(0.9, 0.99)" \
        --adam-eps 1e-6 \
        --clip-norm 1.0 \
        --lr-scheduler polynomial_decay \
        --lr $lr \
        --warmup-ratio $warmup \
        --max-epoch $epoch \
        --batch-size $batch_size \
        --pooler-dropout $dropout \
        --update-freq 1 \
        --seed $seed \
        --save-interval-updates 50 \
        --validate-interval-updates 50 \
        --keep-interval-updates 1 \
        --no-epoch-checkpoints \
        --log-interval 100 \
        --log-format simple \
        --validate-interval 1 \
        --finetune-from-model $weight_path \
        --best-checkpoint-metric $metric \
        --maximize-best-checkpoint-metric \
        --patience 40 \
        --encoder-embed-dim 128 \
        --encoder-attention-heads 8 \
        --encoder-ffn-embed-dim 128 \
        --encoder-layers 10 \
        --decoder-layers 5 \
        --decoder-ffn-embed-dim 128 \
        --decoder-attention-heads 8 \
        --max-seq-len 128 \
        --save-dir $save_dir \
        --tmp-save-dir $save_dir > ${logfile} 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[SUCCESS] GPU${gpu_id} | ${task_name}: lr=${lr}, bs=${batch_size}, drop=${dropout}, warm=${warmup}" | tee -a "gpu${gpu_id}_worker.log"

        if [ "$metric" = "valid_agg_rmse" ]; then
            best_value=$(grep "best_valid_agg_rmse" ${logfile} | tail -1 | awk '{print $NF}')
            [ ! -z "$best_value" ] && echo "lr=${lr}, bs=${batch_size}, dropout=${dropout}, warmup=${warmup} -> best_rmse=${best_value}" >> ${base_save_dir}/results_summary.txt
        else
            best_value=$(grep "best_valid_agg_auc" ${logfile} | tail -1 | awk '{print $NF}')
            [ ! -z "$best_value" ] && echo "lr=${lr}, bs=${batch_size}, dropout=${dropout}, warmup=${warmup} -> best_auc=${best_value}" >> ${base_save_dir}/results_summary.txt
        fi
    else
        echo "[FAILED] GPU${gpu_id} | ${task_name}: lr=${lr}, bs=${batch_size}, drop=${dropout}, warm=${warmup}" | tee -a "gpu${gpu_id}_worker.log"
        echo "lr=${lr}, bs=${batch_size}, dropout=${dropout}, warmup=${warmup} -> FAILED" >> ${base_save_dir}/results_summary.txt
    fi

    sleep 1
}

# Export function for parallel execution
export -f run_experiment
export -f is_experiment_completed
export data_path dict_name weight_path n_gpu arch epoch conf_size seed delta_pair_repr_norm_loss

# Generate all experiment jobs
generate_all_jobs() {
    local output_file=$1
    > "$output_file"

    local job_id=0
    # Dataset loop is now innermost
    for lr in "${learning_rates[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for dropout in "${pooler_dropouts[@]}"; do
                for warmup in "${warmup_ratios[@]}"; do
                    for dataset_info in "${datasets[@]}"; do
                        IFS='|' read -r task_name task_num loss_func metric base_port mem_cat <<< "$dataset_info"

                        job_id=$((job_id + 1))
                        # Assign GPU alternately
                        gpu_id=$(( (job_id % 2) + 2 ))

                        echo "${gpu_id}|${task_name}|${task_num}|${loss_func}|${metric}|${base_port}|${lr}|${batch_size}|${dropout}|${warmup}|${job_id}|${mem_cat}" >> "$output_file"
                    done
                done
            done
        done
    done

    echo "Generated $job_id jobs total"
}

# Worker with controlled parallelism based on memory
worker() {
    local gpu_id=$1
    local job_file=$2
    local max_parallel=$3

    echo "=========================================="
    echo "Starting GPU ${gpu_id} Worker (max ${max_parallel} parallel jobs)"
    echo "=========================================="

    local completed=0
    local skipped=0
    local failed=0

    # Read jobs for this GPU
    local jobs=()
    while IFS='|' read -r gid task_name task_num loss_func metric base_port lr batch_size dropout warmup exp_id mem_cat; do
        if [ "$gid" -eq "$gpu_id" ]; then
            jobs+=("${task_name}|${task_num}|${loss_func}|${metric}|${base_port}|${lr}|${batch_size}|${dropout}|${warmup}|${exp_id}|${mem_cat}")
        fi
    done < "$job_file"

    local total=${#jobs[@]}
    echo "Total jobs for GPU ${gpu_id}: ${total}"

    # Process jobs with controlled parallelism
    local i=0
    while [ $i -lt ${total} ]; do
        # Count running jobs
        local running=$(jobs -r | wc -l)

        # Determine max concurrent based on current job memory category
        IFS='|' read -r task_name task_num loss_func metric base_port lr batch_size dropout warmup exp_id mem_cat <<< "${jobs[$i]}"

        local max_concurrent=$MAX_CONCURRENT_SMALL
        if [ "$mem_cat" = "medium" ]; then
            max_concurrent=$MAX_CONCURRENT_MEDIUM
        elif [ "$mem_cat" = "large" ]; then
            max_concurrent=$MAX_CONCURRENT_LARGE
        fi

        # If batch_size is 64, reduce concurrent jobs
        if [ "$batch_size" -eq 64 ]; then
            max_concurrent=$((max_concurrent - 1))
            [ $max_concurrent -lt 1 ] && max_concurrent=1
        fi

        # Wait if we're at max concurrent
        while [ $(jobs -r | wc -l) -ge $max_concurrent ]; do
            sleep 5
        done

        # Check if already completed
        local base_save_dir="./save_molnet/save_${task_name}"
        local save_dir="${base_save_dir}/lr${lr}_bs${batch_size}_drop${dropout}_warm${warmup}"

        if is_experiment_completed "$save_dir"; then
            skipped=$((skipped + 1))
            echo "[SKIP] GPU${gpu_id} | ${task_name}: lr=${lr}, bs=${batch_size}, drop=${dropout}, warm=${warmup} | Progress: $((i+1))/${total}" | tee -a "gpu${gpu_id}_worker.log"
        else
            # Run in background
            run_experiment $gpu_id "$task_name" "$task_num" "$loss_func" "$metric" "$base_port" "$lr" "$batch_size" "$dropout" "$warmup" "$exp_id" &

            completed=$((completed + 1))
            echo "[LAUNCHED] GPU${gpu_id} | ${task_name}: lr=${lr}, bs=${batch_size}, drop=${dropout}, warm=${warmup} | Progress: $((i+1))/${total} | Parallel: $(($(jobs -r | wc -l)))/${max_concurrent}" | tee -a "gpu${gpu_id}_worker.log"
        fi

        i=$((i + 1))

        # Small delay to prevent overwhelming the system
        sleep 0.5
    done

    # Wait for all background jobs to complete
    echo "Waiting for all background jobs to complete on GPU ${gpu_id}..." | tee -a "gpu${gpu_id}_worker.log"
    wait

    echo ""
    echo "=========================================="
    echo "GPU ${gpu_id} Worker Completed!"
    echo "Completed: ${completed}, Skipped: ${skipped}"
    echo "=========================================="
}

# Main execution
main() {
    echo "=========================================="
    echo "Maximum Parallel Grid Search Runner"
    echo "Concurrent jobs per GPU: Small=4, Medium=3, Large=2"
    echo "=========================================="
    echo ""

    # Validate prerequisites
    if [ ! -d "$data_path" ] || [ ! -f "$weight_path" ]; then
        echo "ERROR: Required files not found"
        exit 1
    fi

    # Count existing completed experiments
    echo "Scanning existing experiments..."
    local total_exp=0
    local completed_exp=0

    # Dataset loop is now innermost
    for lr in "${learning_rates[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for dropout in "${pooler_dropouts[@]}"; do
                for warmup in "${warmup_ratios[@]}"; do
                    for dataset_info in "${datasets[@]}"; do
                        IFS='|' read -r task_name _ _ _ _ _ <<< "$dataset_info"
                        local base_save_dir="./save_${task_name}"

                        total_exp=$((total_exp + 1))
                        local save_dir="${base_save_dir}/lr${lr}_bs${batch_size}_drop${dropout}_warm${warmup}"
                        if is_experiment_completed "$save_dir"; then
                            completed_exp=$((completed_exp + 1))
                        fi
                    done
                done
            done
        done
    done

    echo "Total experiments: ${total_exp}"
    echo "Already completed: ${completed_exp}"
    echo "Remaining: $((total_exp - completed_exp))"
    echo ""

    # Generate job file
    job_file="/tmp/grid_search_max_parallel_jobs_$$.txt"
    generate_all_jobs "$job_file"

    # Start workers
    echo "Starting workers with maximum parallelization..."
    echo ""

    worker 2 "$job_file" 4 &
    pid_gpu2=$!

    worker 3 "$job_file" 4 &
    pid_gpu3=$!

    echo "Workers launched:"
    echo "  GPU 2 Worker PID: ${pid_gpu2}"
    echo "  GPU 3 Worker PID: ${pid_gpu3}"
    echo ""
    echo "Monitor with:"
    echo "  tail -f gpu2_worker.log"
    echo "  tail -f gpu3_worker.log"
    echo "  watch -n 1 nvidia-smi"
    echo ""

    # Wait for completion
    wait $pid_gpu2
    wait $pid_gpu3

    rm -f "$job_file"

    echo ""
    echo "=========================================="
    echo "All Grid Searches Completed!"
    echo "=========================================="
}

# Run main
main
