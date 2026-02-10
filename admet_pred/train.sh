#!/bin/bash

# ==============================================================================
# ADMET Grid Search Runner
# - Runs experiments for TDC ADMET datasets
# - Supports both classification and regression tasks
# - Optimized for GPU memory management
# - Auto-balances small and large datasets to avoid OOM
# ==============================================================================

# Activate conda environment
source /home/csy/anaconda3/etc/profile.d/conda.sh
conda activate lf_cfm

cd /home/csy/work1/3D/TheMol

# Common configuration
data_path="/home/csy/work1/3D/TheMol/molecular_property_prediction/admet_group"
dict_name="dict.txt"
weight_path="/home/csy/work1/3D/TheMol/saveOptimal2_Flow/checkpoint_last.pt"
n_gpu=1
arch=unimol_Optimal_padding2
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
#export CUDA_VISIBLE_DEVICES="2,3"

# Hyperparameter grid
learning_rates=(5e-5 8e-5 1e-4)
batch_sizes=(32 64)
pooler_dropouts=(0.0 0.1 0.2 0.5)
warmup_ratios=(0.0 0.06 0.1)

# ADMET Dataset configurations with memory estimates
# Format: name|task_num|loss_func|metric|base_port|mem_category|train_size
# mem_category: small (<1k), medium (1k-2k), large (2k-5k), xlarge (5k+)
declare -a datasets=(
    # Classification tasks (binary)
    "ames|2|finetune_cross_entropy|valid_agg_auc|10400|xlarge|5820"
    "bbb|2|finetune_cross_entropy|valid_agg_auc|10500|medium|1620"
    "bioav|2|finetune_cross_entropy|valid_agg_auc|10600|small|512"
    "CYP2C9-inhibition|2|finetune_cross_entropy|valid_agg_auprc|10700|xlarge|9560"
    "CYP2C9-substrate|2|finetune_cross_entropy|valid_agg_auprc|10800|small|533"
    "CYP2D6-inhibition|2|finetune_cross_entropy|valid_agg_auprc|10900|xlarge|10398"
    "CYP2D6-substrate|2|finetune_cross_entropy|valid_agg_auprc|11000|small|531"
    "CYP3A4-inhibition|2|finetune_cross_entropy|valid_agg_auprc|11100|xlarge|9752"
    "CYP3A4-substrate|2|finetune_cross_entropy|valid_agg_auprc|11200|small|533"
    "dili|2|finetune_cross_entropy|valid_agg_auc|11300|small|378"
    "herg|2|finetune_cross_entropy|valid_agg_auc|11400|small|516"
    "hia|2|finetune_cross_entropy|valid_agg_auc|11500|small|454"
    "pgp|2|finetune_cross_entropy|valid_agg_auc|11600|small|962"

    # Regression tasks
    "aqsol|1|finetune_smooth_mae|valid_agg_mae|12000|xlarge|7659"
    "caco2|1|finetune_smooth_mae|valid_agg_mae|12100|small|716"
    "cl-hepa|1|finetune_smooth_mae|valid_agg_spearman|12200|small|970"
    "cl-micro|1|finetune_smooth_mae|valid_agg_spearman|12300|small|780"
    "half-life|1|finetune_smooth_mae|valid_agg_spearman|12400|small|452"
    "LD50|1|finetune_smooth_mae|valid_agg_mae|12500|xlarge|5728"
    "lipo|1|finetune_smooth_mae|valid_agg_mae|12600|large|3342"
    "ppbr|1|finetune_smooth_mae|valid_agg_mae|12700|large|2002"
    "vdss|1|finetune_smooth_mae|valid_agg_spearman|12800|small|771"
)

# Maximum concurrent jobs per GPU based on dataset size and memory
# small tasks (<1k): up to 4 concurrent (bs=32) or 3 (bs=64)
# medium tasks (1k-2k): up to 3 concurrent (bs=32) or 2 (bs=64)
# large tasks (2k-5k): up to 2 concurrent (bs=32) or 1 (bs=64)
# xlarge tasks (5k+): up to 2 concurrent (bs=32) or 1 (bs=64)
MAX_CONCURRENT_SMALL=4
MAX_CONCURRENT_MEDIUM=3
MAX_CONCURRENT_LARGE=2
MAX_CONCURRENT_XLARGE=2

# Function to check if experiment is already completed
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

    # Changed: Save to /home/csy/work1/3D/TheMol/save_tdc/save_*
    local base_save_dir="./save_tdc/save_${task_name}"
    local save_dir="${base_save_dir}/lr${lr}_bs${batch_size}_drop${dropout}_warm${warmup}"
    local logfile="${save_dir}/${arch}.log"

    # Check if already completed
    if is_experiment_completed "$save_dir"; then
        return 0
    fi

    mkdir -p "${save_dir}"

    # Calculate unique port
    local MASTER_PORT
    if [ $gpu_id -eq 1 ]; then
        MASTER_PORT=$((base_port + exp_id))
    elif [ $gpu_id -eq 2 ]; then
        MASTER_PORT=$((base_port + 1000 + exp_id))
    else
        MASTER_PORT=$((base_port + 2000 + exp_id))
    fi

    echo "[START] GPU${gpu_id} | ${task_name}: lr=${lr}, bs=${batch_size}, drop=${dropout}, warm=${warmup} | Port: ${MASTER_PORT}" | tee -a "gpu${gpu_id}_worker.log"

    # Determine if metric should be maximized or minimized
    # MAE: lower is better (minimize) - no --maximize flag
    # AUC, AUPRC, Spearman: higher is better (maximize) - add --maximize flag
    if [ "$metric" = "valid_agg_mae" ]; then
        maximize_flag=""
    else
        maximize_flag="--maximize-best-checkpoint-metric"
    fi

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
        --save-interval-updates 10 \
        --validate-interval-updates 10 \
        --keep-interval-updates 1 \
        --no-epoch-checkpoints \
        --log-interval 10 \
        --log-format simple \
        --validate-interval 1 \
        --finetune-from-model $weight_path \
        --best-checkpoint-metric $metric \
        $maximize_flag \
        --patience 100 \
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

        if [ "$metric" = "valid_agg_mae" ]; then
            best_value=$(grep "best_valid_agg_mae" ${logfile} | tail -1 | awk '{print $NF}')
            [ ! -z "$best_value" ] && echo "lr=${lr}, bs=${batch_size}, dropout=${dropout}, warmup=${warmup} -> best_mae=${best_value}" >> ${base_save_dir}/results_summary.txt
        elif [ "$metric" = "valid_agg_spearman" ]; then
            best_value=$(grep "best_valid_agg_spearman" ${logfile} | tail -1 | awk '{print $NF}')
            [ ! -z "$best_value" ] && echo "lr=${lr}, bs=${batch_size}, dropout=${dropout}, warmup=${warmup} -> best_spearman=${best_value}" >> ${base_save_dir}/results_summary.txt
        elif [ "$metric" = "valid_agg_auprc" ]; then
            best_value=$(grep "best_valid_agg_auprc" ${logfile} | tail -1 | awk '{print $NF}')
            [ ! -z "$best_value" ] && echo "lr=${lr}, bs=${batch_size}, dropout=${dropout}, warmup=${warmup} -> best_auprc=${best_value}" >> ${base_save_dir}/results_summary.txt
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
    local gpu_list=$2
    > "$output_file"

    local job_id=0
    # for dataset_info in "${datasets[@]}"; do
    #     IFS='|' read -r task_name task_num loss_func metric base_port mem_cat train_size <<< "$dataset_info"

    for lr in "${learning_rates[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for dropout in "${pooler_dropouts[@]}"; do
                for warmup in "${warmup_ratios[@]}"; do
                    for dataset_info in "${datasets[@]}"; do
                        IFS='|' read -r task_name task_num loss_func metric base_port mem_cat train_size <<< "$dataset_info"

                        job_id=$((job_id + 1))

                        # Assign GPU based on gpu_list
                        local gpu_count=$(echo $gpu_list | tr ',' ' ' | wc -w)
                        local gpu_array=($(echo $gpu_list | tr ',' ' '))
                        local gpu_id=${gpu_array[$((job_id % gpu_count))]}

                        echo "${gpu_id}|${task_name}|${task_num}|${loss_func}|${metric}|${base_port}|${lr}|${batch_size}|${dropout}|${warmup}|${job_id}|${mem_cat}|${train_size}" >> "$output_file"
                    done
                done
            done
        done
    done

    echo "Generated $job_id jobs total for GPUs: $gpu_list"
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
    while IFS='|' read -r gid task_name task_num loss_func metric base_port lr batch_size dropout warmup exp_id mem_cat train_size; do
        if [ "$gid" -eq "$gpu_id" ]; then
            jobs+=("${task_name}|${task_num}|${loss_func}|${metric}|${base_port}|${lr}|${batch_size}|${dropout}|${warmup}|${exp_id}|${mem_cat}|${train_size}")
        fi
    done < "$job_file"

    local total=${#jobs[@]}
    echo "Total jobs for GPU ${gpu_id}: ${total}"

    # Process jobs with controlled parallelism
    local i=0
    while [ $i -lt ${total} ]; do
        # Count running jobs
        local running=$(jobs -r | wc -l)

        # Determine max concurrent based on current job memory category and batch size
        IFS='|' read -r task_name task_num loss_func metric base_port lr batch_size dropout warmup exp_id mem_cat train_size <<< "${jobs[$i]}"

        local max_concurrent=$MAX_CONCURRENT_SMALL
        if [ "$mem_cat" = "medium" ]; then
            max_concurrent=$MAX_CONCURRENT_MEDIUM
        elif [ "$mem_cat" = "large" ]; then
            max_concurrent=$MAX_CONCURRENT_LARGE
        elif [ "$mem_cat" = "xlarge" ]; then
            max_concurrent=$MAX_CONCURRENT_XLARGE
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
        local base_save_dir="./save_tdc/save_${task_name}"
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
    # Parse command line arguments for GPU selection
    local GPU_LIST="2,3"  # Default GPUs

    if [ $# -gt 0 ]; then
        GPU_LIST="$1"
    fi

    echo "=========================================="
    echo "ADMET Grid Search Runner"
    echo "Using GPUs: ${GPU_LIST}"
    echo "Concurrent jobs per GPU: Small=4, Medium=3, Large=2, XLarge=2"
    echo "Save directory: ./save_tdc/save_*"
    echo "=========================================="
    echo ""

    # Validate prerequisites
    if [ ! -d "$data_path" ] || [ ! -f "$weight_path" ]; then
        echo "ERROR: Required files not found"
        echo "  Data path: $data_path"
        echo "  Weight path: $weight_path"
        exit 1
    fi

    # Create save_tdc directory
    mkdir -p ./save_tdc

    # Count existing completed experiments
    echo "Scanning existing experiments..."
    local total_exp=0
    local completed_exp=0

    for dataset_info in "${datasets[@]}"; do
        IFS='|' read -r task_name _ _ _ _ _ _ <<< "$dataset_info"
        local base_save_dir="./save_tdc/save_${task_name}"

        for lr in "${learning_rates[@]}"; do
            for batch_size in "${batch_sizes[@]}"; do
                for dropout in "${pooler_dropouts[@]}"; do
                    for warmup in "${warmup_ratios[@]}"; do
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
    job_file="/tmp/admet_grid_search_jobs_$$.txt"
    generate_all_jobs "$job_file" "$GPU_LIST"

    # Start workers based on GPU list
    echo "Starting workers with maximum parallelization..."
    echo ""

    local gpu_array=($(echo $GPU_LIST | tr ',' ' '))
    local pids=()

    for gpu_id in "${gpu_array[@]}"; do
        worker $gpu_id "$job_file" 4 &
        pids+=($!)
        echo "  GPU ${gpu_id} Worker PID: ${pids[-1]}"
    done

    echo ""
    echo "Monitor with:"
    for gpu_id in "${gpu_array[@]}"; do
        echo "  tail -f gpu${gpu_id}_worker.log"
    done
    echo "  watch -n 1 nvidia-smi"
    echo ""

    # Wait for completion
    for pid in "${pids[@]}"; do
        wait $pid
    done

    rm -f "$job_file"

    echo ""
    echo "=========================================="
    echo "All Grid Searches Completed!"
    echo "=========================================="
}

# Run main
main "$@"
