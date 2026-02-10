#!/bin/bash

echo "========================================"
echo "ADMET Grid Search Results Summary"
echo "========================================"
echo ""

# Hyperparameter grid (from run_admet_grid_search.sh)
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

# List of ADMET dataset directories
datasets=(
    # Classification tasks (binary)
    "save_tdc/save_ames"
    "save_tdc/save_bbb"
    "save_tdc/save_bioav"
    "save_tdc/save_CYP2C9-inhibition"
    "save_tdc/save_CYP2C9-substrate"
    "save_tdc/save_CYP2D6-inhibition"
    "save_tdc/save_CYP2D6-substrate"
    "save_tdc/save_CYP3A4-inhibition"
    "save_tdc/save_CYP3A4-substrate"
    "save_tdc/save_dili"
    "save_tdc/save_herg"
    "save_tdc/save_hia"
    "save_tdc/save_pgp"

    # Regression tasks
    "save_tdc/save_aqsol"
    "save_tdc/save_caco2"
    "save_tdc/save_cl-hepa"
    "save_tdc/save_cl-micro"
    "save_tdc/save_half-life"
    "save_tdc/save_LD50"
    "save_tdc/save_lipo"
    "save_tdc/save_ppbr"
    "save_tdc/save_vdss"
)

# Define task types for each dataset
# classification: higher is better (AUC)
# regression: lower is better (RMSE)
declare -A task_types
# Classification tasks
task_types["save_tdc/save_ames"]="classification"
task_types["save_tdc/save_bbb"]="classification"
task_types["save_tdc/save_bioav"]="classification"
task_types["save_tdc/save_CYP2C9-inhibition"]="classification"
task_types["save_tdc/save_CYP2C9-substrate"]="classification"
task_types["save_tdc/save_CYP2D6-inhibition"]="classification"
task_types["save_tdc/save_CYP2D6-substrate"]="classification"
task_types["save_tdc/save_CYP3A4-inhibition"]="classification"
task_types["save_tdc/save_CYP3A4-substrate"]="classification"
task_types["save_tdc/save_dili"]="classification"
task_types["save_tdc/save_herg"]="classification"
task_types["save_tdc/save_hia"]="classification"
task_types["save_tdc/save_pgp"]="classification"

# Regression tasks
task_types["save_tdc/save_aqsol"]="regression"
task_types["save_tdc/save_caco2"]="regression"
task_types["save_tdc/save_cl-hepa"]="regression"
task_types["save_tdc/save_cl-micro"]="regression"
task_types["save_tdc/save_half-life"]="regression"
task_types["save_tdc/save_LD50"]="regression"
task_types["save_tdc/save_lipo"]="regression"
task_types["save_tdc/save_ppbr"]="regression"
task_types["save_tdc/save_vdss"]="regression"

# Define metric names for each dataset (from run_admet_grid_search.sh)
declare -A metric_names
# AUC metrics
metric_names["ames"]="best_auc"
metric_names["bbb"]="best_auc"
metric_names["bioav"]="best_auc"
metric_names["dili"]="best_auc"
metric_names["herg"]="best_auc"
metric_names["hia"]="best_auc"
metric_names["pgp"]="best_auc"
# AUPRC metrics
metric_names["CYP2C9-inhibition"]="best_auprc"
metric_names["CYP2C9-substrate"]="best_auprc"
metric_names["CYP2D6-inhibition"]="best_auprc"
metric_names["CYP2D6-substrate"]="best_auprc"
metric_names["CYP3A4-inhibition"]="best_auprc"
metric_names["CYP3A4-substrate"]="best_auprc"
# MAE metrics
metric_names["aqsol"]="best_mae"
metric_names["caco2"]="best_mae"
metric_names["LD50"]="best_mae"
metric_names["lipo"]="best_mae"
metric_names["ppbr"]="best_mae"
# Spearman metrics
metric_names["cl-hepa"]="best_spearman"
metric_names["cl-micro"]="best_spearman"
metric_names["half-life"]="best_spearman"
metric_names["vdss"]="best_spearman"

# TDC SOTA scores mapping
declare -A tdc_sota
tdc_sota["caco2"]="0.256"
tdc_sota["hia"]="0.993"
tdc_sota["pgp"]="0.938"
tdc_sota["bioav"]="0.942"
tdc_sota["lipo"]="0.456"
tdc_sota["aqsol"]="0.741"
tdc_sota["bbb"]="0.924"
tdc_sota["ppbr"]="7.440"
tdc_sota["vdss"]="0.713"
tdc_sota["CYP2D6-inhibition"]="0.790"
tdc_sota["CYP3A4-inhibition"]="0.916"
tdc_sota["CYP2C9-inhibition"]="0.859"
tdc_sota["CYP2D6-substrate"]="0.736"
tdc_sota["CYP3A4-substrate"]="0.667"
tdc_sota["CYP2C9-substrate"]="0.474"
tdc_sota["half-life"]="0.576"
tdc_sota["cl-micro"]="0.630"
tdc_sota["cl-hepa"]="0.536"
tdc_sota["herg"]="0.880"
tdc_sota["ames"]="0.871"
tdc_sota["dili"]="0.956"
tdc_sota["LD50"]="0.552"

# Initialize totals
total_experiments=0
total_success=0
total_failed=0
num_active_datasets=0

for dataset_dir in "${datasets[@]}"; do
    full_path="/home/csy/work1/3D/TheMol/${dataset_dir}"

    # Check if directory exists
    if [ ! -d "$full_path" ]; then
        continue
    fi

    # Count this as an active dataset
    num_active_datasets=$((num_active_datasets + 1))

    # Extract dataset name (remove save_tdc/save_ prefix)
    dataset_name=$(echo "$dataset_dir" | sed 's|save_tdc/save_||')
    dataset_name_upper=$(echo "$dataset_name" | tr '[:lower:]' '[:upper:]')

    # Count successful and failed experiments from directory structure
    success=0
    failed=0

    # Iterate through all hyperparameter combinations
    for lr in "${learning_rates[@]}"; do
        for bs in "${batch_sizes[@]}"; do
            for dropout in "${pooler_dropouts[@]}"; do
                for warmup in "${warmup_ratios[@]}"; do
                    # Construct directory name
                    exp_dir="${full_path}/lr${lr}_bs${bs}_drop${dropout}_warm${warmup}"

                    if [ -d "$exp_dir" ]; then
                        # Check if experiment is successful (checkpoint_best.pt exists)
                        if [ -f "${exp_dir}/checkpoint_best.pt" ]; then
                            success=$((success + 1))
                        else
                            # Check if it's actually failed or just not started
                            if [ -f "${exp_dir}/unimol_Optimal_padding.log" ]; then
                                failed=$((failed + 1))
                            fi
                        fi
                    fi
                done
            done
        done
    done

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

        # Get the metric name for this dataset
        metric_name="${metric_names[$dataset_name]}"

        # Determine sort order based on metric type
        # MAE: lower is better (ascending)
        # AUC, AUPRC, Spearman: higher is better (descending)
        if [ "$metric_name" = "best_mae" ]; then
            sort_option="-n"
        else
            sort_option="-nr"
        fi

        grep "$metric_name" "$summary_file" 2>/dev/null | \
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

# Show best result from each dataset with TDC SOTA
echo "========================================"
echo "Best Result per Dataset (vs TDC SOTA)"
echo "========================================"
for dataset_dir in "${datasets[@]}"; do
    full_path="/home/csy/work1/3D/TheMol/${dataset_dir}"
    summary_file="${full_path}/results_summary.txt"

    if [ ! -f "$summary_file" ]; then
        continue
    fi

    dataset_name=$(echo "$dataset_dir" | sed 's|save_tdc/save_||')
    dataset_name_upper=$(echo "$dataset_name" | tr '[:lower:]' '[:upper:]')

    # Get the metric name for this dataset
    metric_name="${metric_names[$dataset_name]}"

    # Determine sort order based on metric type
    # MAE: lower is better (ascending)
    # AUC, AUPRC, Spearman: higher is better (descending)
    if [ "$metric_name" = "best_mae" ]; then
        sort_option="-n"
        metric_type="(lower is better)"
    else
        sort_option="-nr"
        metric_type="(higher is better)"
    fi

    # Get best result
    best=$(grep "$metric_name" "$summary_file" 2>/dev/null | sort -t'=' -k6 $sort_option | head -1)

    if [ ! -z "$best" ]; then
        # Get TDC SOTA score
        sota_score="${tdc_sota[$dataset_name]}"
        if [ -z "$sota_score" ]; then
            sota_score="N/A"
        fi

        echo "$dataset_name_upper $metric_type: $best | TDC_SOTA=$sota_score"
    fi
done
echo ""
