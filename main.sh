#!/bin/bash
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200   # or longer
export TORCH_NCCL_ENABLE_MONITORING=0 

#----------------------------------------Model and Benchmark Settings----------------------------------------#
# Devices to use
devices="0 1 2 3 4 5 6 7"

# '0' for training, '1' for testing
test=1

# For test, lightning saved version
# The first of the 5-fold versions to test, e.g., if trained versions are 20,21,22,23,24, set version=20
version=5
# (optional) a previous 5-fold result for comparison
original_version=90

# Benchmark to train/test; choose from:
# benchmark='posex'
# benchmark="astex_posex"
benchmark="moad"
# benchmark="posex_self_docking"
# benchmark="posex_cross_docking"
# benchmark='astex'
# benchmark='posebusters'

# Relaxation setting for the benchmark, 'both' for including both relaxed and unrelaxed poses
relaxation='both' # 'both', 'true', or 'false'

# Model to train
model='MolAS'

# Jump per-fold evaluation and use saved results (can activate after testing once)
jump_eval='false'

seed=42
#------------------------------------------------------------------------------------------------------------#

num_devices=$(echo $devices | wc -w)

declare -A num_classes_both_dict=(
  [posex]=48
  [astex]=48
  [astex_posex]=48
  [moad]=8
  [posex_self_docking]=48
  [posex_cross_docking]=48
  [posebusters]=14
  [posebench]=14
)
declare -A num_classes_dict=(
  [posex]=24
  [astex]=24
  [astex_posex]=24
  [moad]=8
  [posex_self_docking]=24
  [posex_cross_docking]=24
  [posebusters]=7
  [posebench]=7
)
declare -A incl_columns_dict=(
  [posex]='surfdock unimol diffdock diffdock_l gnina autodock glide alphafold3 moe dynamicbind rfaa fabind equibind boltz1x tankbind neuralplexer chai deepdock boltz ifd protenix diffdock_pocket DS interformer'
  [astex]='surfdock unimol diffdock diffdock_l gnina autodock glide alphafold3 moe dynamicbind rfaa fabind equibind boltz1x tankbind neuralplexer chai deepdock boltz ifd protenix diffdock_pocket DS interformer'
  [astex_posex]='surfdock unimol diffdock diffdock_l gnina autodock glide alphafold3 moe dynamicbind rfaa fabind equibind boltz1x tankbind neuralplexer chai deepdock boltz ifd protenix diffdock_pocket DS interformer'
  [moad]=''
  [posex_self_docking]='surfdock unimol diffdock diffdock_l gnina autodock glide alphafold3 moe dynamicbind rfaa fabind equibind boltz1x tankbind neuralplexer chai deepdock boltz ifd protenix diffdock_pocket DS interformer'
  [posex_cross_docking]='surfdock unimol diffdock diffdock_l gnina autodock glide alphafold3 moe dynamicbind rfaa fabind equibind boltz1x tankbind neuralplexer chai deepdock boltz ifd protenix diffdock_pocket DS interformer'
  [posebusters]='deepdock diffdock equibind gold tankbind unimol autodock'
)

echo "Number of classes for benchmark $benchmark: $num_classes"

if [ "$relaxation" == "both" ]; then
  incl_columns=''
  num_classes=${num_classes_both_dict[$benchmark]}
elif [ "$relaxation" == "true" ]; then
  # Add '_relaxation' suffix to each method in incl_columns
  # incl_columns="${incl_columns_dict[$benchmark]}"_relaxation only adds suffix to the last method
  incl_columns=""
  for method in ${incl_columns_dict[$benchmark]}; do
    incl_columns+="${method}_relaxation "
  done
  echo "Included columns with relaxation: $incl_columns"
  num_classes=${num_classes_dict[$benchmark]}
else
  incl_columns="${incl_columns_dict[$benchmark]}"
  num_classes=${num_classes_dict[$benchmark]}
fi

log_root="lightning_logs"
log_name="benchmark_experiment_${benchmark}"

if [ "$model" == "MCGNNASDock" ]; then
  num_node_features=960
else
  num_node_features=1152
fi

if [ "$test" -eq 0 ]; then
  for i in {0..4}; do
    echo "Training with fold $i"
    torchrun --standalone --nproc_per_node=$num_devices main.py \
        --k_folds 5 \
        --fold_num $i \
        --num_node_features $num_node_features \
        --max_epochs 300 \
        --devices $devices \
        --num_classes $num_classes \
        --incl_columns $incl_columns \
        --batch_size 16 \
        --num_workers 0 \
        --lr 1e-3 \
        --benchmark $benchmark \
        --seed $seed \
        --test_size 0.2 \
        --log_root $log_root \
        --log_name $log_name \
        --model $model
  done
else
  original_ckpts=()
  after_ckpts=()
  for i in {0..4}; do
    echo "Testing with fold $i"

    # Detect .ckpt files starting with 'best-epoch=xxx' with largest xxx
    dir="${log_root}/${log_name}/version_${version}/checkpoints"
    original_dir="${log_root}/${log_name}/version_${original_version}/checkpoints"

    # Find all files matching the pattern
    best_ckpt=$(find "$dir" -type f -name 'best-epoch=*-*.ckpt' | \
    grep -oE 'best-epoch=([0-9]+)-[^/]*\.ckpt$' | \
    sed -E 's/^best-epoch=([0-9]+)-.*/\1 &/' | \
    sort -nr | \
    head -n1 | \
    cut -d' ' -f2-)

    original_best_ckpt=$(find "$original_dir" -type f -name 'best-epoch=*-*.ckpt' | \
    grep -oE 'best-epoch=([0-9]+)-[^/]*\.ckpt$' | \
    sed -E 's/^best-epoch=([0-9]+)-.*/\1 &/' | \
    sort -nr | \
    head -n1 | \
    cut -d' ' -f2-)

    best_ckpt_path="$dir/$best_ckpt"
    original_best_ckpt_path="${log_root}/benchmark_experiment_astex_posex/version_${original_version}/checkpoints/$original_best_ckpt"

    original_ckpts+=("$original_best_ckpt_path")
    after_ckpts+=("$best_ckpt_path") 

    # jump_eval='true' to enable jump evaluation
    if [ "$jump_eval" == "false" ]; then
      echo "#---------------------------Testing with checkpoint: $best_ckpt_path---------------------------#"
      python main.py  \
          --k_folds 5 \
          --fold_num $i \
          --num_node_features $num_node_features \
          --devices $devices \
          --num_classes $num_classes \
          --incl_columns $incl_columns \
          --num_workers 1 \
          --benchmark $benchmark \
          --seed $seed \
          --test_size 0.2 \
          --test \
          --ckpt_path $best_ckpt_path \
          --model $model
      echo "Finished testing with checkpoint: $best_ckpt_path"
    fi
    
    version=$((version + 1))
    original_version=$((original_version + 1))
  done
  
  python stats_test.py \
      --original_ckpts "${original_ckpts[@]}" \
      --after_ckpts "${after_ckpts[@]}"
fi