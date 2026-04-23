export WANDB_API_KEY=35ed06feafa826b6d6dd0c186d59eeba150e7442
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# export MOLMO_LOG_MEMORY=1
export MLSPACES_MAX_HOUSES_PER_WORKER=5
export MLSPACES_JSON_EVAL_EPISODES_PER_BATCH=3      # 每个 batch 跑多少个 episode（控制内存）
export MLSPACES_APPEND_TO_SINGLE_BATCH_FILE=1 

## Pick
# python molmo_spaces/evaluation/eval_main.py   \
#  molmo_spaces.evaluation.configs.evaluation_configs:WallXJointServerAdapterEvalConfig \
#  --benchmark_dir assets/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231/ \
#  --task_horizon_steps 450 \
#  --num_workers 5

# ## pick and place
# python molmo_spaces/evaluation/eval_main.py   \
#  molmo_spaces.evaluation.configs.evaluation_configs:WallXJointServerAdapterEvalConfig \
#  --benchmark_dir assets/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark \
#  --task_horizon_steps 450 \
#  --num_workers 5

# ## open
# python molmo_spaces/evaluation/eval_main.py   \
#  molmo_spaces.evaluation.configs.evaluation_configs:WallXJointServerAdapterEvalConfig \
#  --benchmark_dir assets/benchmarks/molmospaces-bench-v1/ithor/FrankaOpenDataGenConfig/FrankaOpenDataGenConfig_20260123_json_benchmark \
#  --task_horizon_steps 450 \
#  --num_workers 5

# close
python molmo_spaces/evaluation/eval_main.py   \
 molmo_spaces.evaluation.configs.evaluation_configs:WallXJointServerAdapterEvalConfig \
 --benchmark_dir assets/benchmarks/molmospaces-bench-v1/ithor/FrankaCloseDataGenConfig/FrankaCloseDataGenConfig_20260123_json_benchmark \
 --task_horizon_steps 450 \
 --num_workers 2

