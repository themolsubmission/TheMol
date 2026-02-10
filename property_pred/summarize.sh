#!/bin/bash

echo "========================================"
echo "Grid Search Results Summary"
echo "========================================"
echo ""

# Hyperparameter grid (from run_all_grid_search_max_parallel.sh)
learning_rates=(5e-5 8e-5 1e-4)
batch_sizes=(32 64)
pooler_dropouts=(0.0 0.1 0.2 0.5)
warmup_ratios=(0.0 0.06 0.1)

# Calculate total hyperparameter combinations
num_lr=${#learning_rates[@]}
num_bs=${#batch_sizes[@]}
num_dropout=${#pooler_dropouts[@]}
num_warmup=${#warmup_ratios[@]}
total_hp_combinations=$((num_lr * num_bs * num_dropout * num_warmup))

echo "Hyperparameter Grid:"
echo "  Learning rates: ${learning_rates[@]} (${num_lr})"
echo "  Batch sizes: ${batch_sizes[@]} (${num_bs})"
echo "  Pooler dropouts: ${pooler_dropouts[@]} (${num_dropout})"
echo "  Warmup ratios: ${warmup_ratios[@]} (${num_warmup})"
echo "  Total combinations: ${total_hp_combinations}"
echo ""

# List of dataset directories
datasets=(
    "save_bace"
    "save_bbbp"
    "save_clintox"
    "save_freesolv"
    "save_hiv"
    "save_lipo"
    "save_muv"
    "save_pcba"
    "save_sider"
    "save_tox21"
    "save_toxcast"
)

# Define task types for each dataset
# classification: higher is better (AUC)
# regression: lower is better (RMSE, MAE)
declare -A task_types
task_types["save_bace"]="classification"
task_types["save_bbbp"]="classification"
task_types["save_clintox"]="classification"
task_types["save_freesolv"]="regression"
task_types["save_hiv"]="classification"
task_types["save_lipo"]="regression"
task_types["save_muv"]="classification"
task_types["save_pcba"]="classification"
task_types["save_sider"]="classification"
task_types["save_tox21"]="classification"
task_types["save_toxcast"]="classification"

# Initialize totals
total_experiments=0
total_success=0
total_failed=0
num_active_datasets=0

# First, count active datasets
for dataset_dir in "${datasets[@]}"; do
    full_path="/home/csy/work1/3D/TheMol/${dataset_dir}"
    if [ -d "$full_path" ]; then
        num_active_datasets=$((num_active_datasets + 1))
    fi
done

# Initialize per-dataset counters using associative arrays
declare -A dataset_success
declare -A dataset_failed
for dataset_dir in "${datasets[@]}"; do
    dataset_success[$dataset_dir]=0
    dataset_failed[$dataset_dir]=0
done

# Iterate through all hyperparameter combinations (dataset loop is innermost)
for lr in "${learning_rates[@]}"; do
    for bs in "${batch_sizes[@]}"; do
        for dropout in "${pooler_dropouts[@]}"; do
            for warmup in "${warmup_ratios[@]}"; do
                for dataset_dir in "${datasets[@]}"; do
                    full_path="/home/csy/work1/3D/TheMol/${dataset_dir}"

                    # Check if directory exists
                    if [ ! -d "$full_path" ]; then
                        continue
                    fi

                    # Construct directory name
                    exp_dir="${full_path}/lr${lr}_bs${bs}_drop${dropout}_warm${warmup}"

                    if [ -d "$exp_dir" ]; then
                        # Check if experiment is successful (checkpoint_best.pt exists)
                        if [ -f "${exp_dir}/checkpoint_best.pt" ]; then
                            dataset_success[$dataset_dir]=$((${dataset_success[$dataset_dir]} + 1))
                        else
                            # Check if it's actually failed or just not started
                            if [ -f "${exp_dir}/unimol_Optimal_padding.log" ]; then
                                dataset_failed[$dataset_dir]=$((${dataset_failed[$dataset_dir]} + 1))
                            fi
                        fi
                    fi
                done
            done
        done
    done
done

# Print results per dataset
for dataset_dir in "${datasets[@]}"; do
    full_path="/home/csy/work1/3D/TheMol/${dataset_dir}"

    # Check if directory exists
    if [ ! -d "$full_path" ]; then
        continue
    fi

    # Extract dataset name (remove save_ prefix)
    dataset_name=$(echo "$dataset_dir" | sed 's/save_//')
    dataset_name_upper=$(echo "$dataset_name" | tr '[:lower:]' '[:upper:]')

    success=${dataset_success[$dataset_dir]}
    failed=${dataset_failed[$dataset_dir]}
    total=$((success + failed))

    # Update global totals
    total_experiments=$((total_experiments + total))
    total_success=$((total_success + success))
    total_failed=$((total_failed + failed))

    # Calculate success rate
    if [ $total -gt 0 ]; then
        success_rate=$(awk "BEGIN {printf \"%.1f\", ($success * 100.0 / $total)}")
    else
        success_rate="0.0"
    fi

    echo "=== $dataset_name_upper ==="
    echo "  Total experiments: $total / ${total_hp_combinations}"
    echo "  Successful: $success"
    echo "  Failed: $failed"
    if [ $total -gt 0 ]; then
        echo "  Success rate: ${success_rate}%"
    fi

    # Show top 5 best results from summary file if it exists
    summary_file="${full_path}/results_summary.txt"
    if [ -f "$summary_file" ] && [ $success -gt 0 ]; then
        echo "  Top 5 results:"

        # Determine sort order based on task type
        task_type="${task_types[$dataset_dir]}"
        if [ "$task_type" = "regression" ]; then
            # For regression: lower is better (ascending order)
            sort_option="-n"
        else
            # For classification: higher is better (descending order)
            sort_option="-nr"
        fi

        grep "best_auc\|best_rmse\|best_mae" "$summary_file" 2>/dev/null | \
            sort -t'=' -k6 $sort_option | head -5 | \
            while IFS= read -r line; do
                echo "    $line"
            done
    fi

    echo ""
done

echo "========================================"
echo "Overall Summary"
echo "========================================"
echo "Active datasets: $num_active_datasets"
echo "Expected total experiments: $((num_active_datasets * total_hp_combinations))"
echo "Actual experiments run: $total_experiments"
echo "Successful: $total_success"
echo "Failed: $total_failed"
echo "Not started: $((num_active_datasets * total_hp_combinations - total_experiments))"
if [ $total_experiments -gt 0 ]; then
    overall_success_rate=$(awk "BEGIN {printf \"%.1f\", ($total_success * 100.0 / $total_experiments)}")
    echo "Overall success rate: ${overall_success_rate}%"
fi
if [ $((num_active_datasets * total_hp_combinations)) -gt 0 ]; then
    completion_rate=$(awk "BEGIN {printf \"%.1f\", ($total_experiments * 100.0 / ($num_active_datasets * $total_hp_combinations))}")
    echo "Completion rate: ${completion_rate}%"
fi
echo ""

# Show best result from each dataset
echo "========================================"
echo "Best Result per Dataset"
echo "========================================"
for dataset_dir in "${datasets[@]}"; do
    full_path="/home/csy/work1/3D/TheMol/${dataset_dir}"
    summary_file="${full_path}/results_summary.txt"

    if [ ! -f "$summary_file" ]; then
        continue
    fi

    dataset_name=$(echo "$dataset_dir" | sed 's/save_//')
    dataset_name_upper=$(echo "$dataset_name" | tr '[:lower:]' '[:upper:]')

    # Determine sort order based on task type
    task_type="${task_types[$dataset_dir]}"
    if [ "$task_type" = "regression" ]; then
        # For regression: lower is better (ascending order)
        sort_option="-n"
        metric_type="(lower is better)"
    else
        # For classification: higher is better (descending order)
        sort_option="-nr"
        metric_type="(higher is better)"
    fi

    # Get best result
    best=$(grep "best_auc\|best_rmse\|best_mae" "$summary_file" 2>/dev/null | sort -t'=' -k6 $sort_option | head -1)

    if [ ! -z "$best" ]; then
        echo "$dataset_name_upper $metric_type: $best"
    fi
done
echo ""
