# Data generation for the four benchmarks referenced in eval_wallx.sh.
# 运行方式：取消下方某一段的注释，然后 `bash datagen_wallx.sh`。
# 输出：统一落到仓库根目录下 data_gen/<TaskTag>/<ExpConfigClassName>/<timestamp>/
#       （main.py 会自动追加 <ExpConfigClassName>/<timestamp> 两层）

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=.
# export MOLMO_LOG_MEMORY=1

# 数据生成阶段沿用 eval 时控制内存的开关（按需保留/删除）。
export MLSPACES_MAX_HOUSES_PER_WORKER=5

NUM_WORKERS=2                  # 数据生成可以用比 eval 更多的并行 worker。
TASK_HORIZON=450               # 与 eval_wallx.sh 保持一致。

# 所有任务统一写到这个根目录，便于训练时一并扫描。
DATA_ROOT=data_gen
# 每个 house 要收集的轨迹条数由 config 里 task_sampler_config.samples_per_house 控制
#（当前为 4，见 object_manipulation_datagen_configs 里各 Franka*DataGen / *MiniBench）。

############################################################
# 1) Pick (procthor-10k val, FrankaOmniPurpose 相机)
#    对应 eval_wallx.sh 里 FrankaPickDroidMiniBench
############################################################
python -m molmo_spaces.data_generation.main \
  molmo_spaces.data_generation.config.object_manipulation_datagen_configs:FrankaPickDroidMiniBench \
  --num_workers ${NUM_WORKERS} \
  --task_horizon ${TASK_HORIZON} \
  --output_dir ${DATA_ROOT}/pick

############################################################
# 2) Pick-and-Place (procthor-10k val)
#    注意注册名是 FrankaPickandPlaceMiniBench（类名是 FrankaPickandPlaceDroidMiniBench）
############################################################
python -m molmo_spaces.data_generation.main \
  molmo_spaces.data_generation.config.object_manipulation_datagen_configs:FrankaPickandPlaceMiniBench \
  --num_workers ${NUM_WORKERS} \
  --task_horizon ${TASK_HORIZON} \
  --output_dir ${DATA_ROOT}/pick_and_place

############################################################
# 3) Open (ithor)
#    config 默认 data_split=train；如果想要 val 集（贴近 eval 用的那份），
#    要么改 config 默认值，要么把 `--data_split val` 加到命令里。
############################################################
python -m molmo_spaces.data_generation.main \
  molmo_spaces.data_generation.config.object_manipulation_datagen_configs:FrankaOpenDataGenConfig \
  --num_workers ${NUM_WORKERS} \
  --task_horizon ${TASK_HORIZON} \
  --data_split train \
  --output_dir ${DATA_ROOT}/open

############################################################
# 4) Close (ithor) — 当前激活
############################################################
python -m molmo_spaces.data_generation.main \
  molmo_spaces.data_generation.config.object_manipulation_datagen_configs:FrankaCloseDataGenConfig \
  --num_workers ${NUM_WORKERS} \
  --task_horizon ${TASK_HORIZON} \
  --data_split train \
  --output_dir ${DATA_ROOT}/close


