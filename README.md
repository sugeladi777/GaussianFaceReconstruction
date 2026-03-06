# RGB_Recon

本仓库用于人脸/头部场景重建：视频拆帧、预处理、COLMAP 位姿估计、2DGS 重建、纹理优化。

## 1. 当前编排架构

- 统一入口：`pipelines/pipeline_runner.py`
- 分步脚本：`pipelines/steps/`
  - `step_frames.py`
  - `step_preprocess.py`
  - `step_colmap.py`
  - `step_2dgs.py`
  - `step_texture.py`

说明：`pipelines/geo/`、`pipelines/full_pipeline.py`、`pipelines/geo_recon.py` 已移除，相关逻辑已合并到 `steps`。

## 2. 流程说明

```text
视频/图片
  -> step_frames（可跳过）
  -> step_preprocess（可跳过）
  -> step_colmap（可跳过）
  -> step_2dgs（可跳过，内含导出 2dgs_recon.obj + transforms.json）
  -> step_texture（可跳过）
```

`step_colmap.py` 默认 `--use-mask 0`（不使用 mask），可通过 `--use_colmap_mask 1` 开启。

## 3. 常用命令

### 视频目录一键全流程

```bash
python pipelines/pipeline_runner.py \
  --video_dir /home/lichengkai/RGB_Recon/input/test \
  --data_root /home/lichengkai/RGB_Recon/output \
  --gpus auto \
  --workers 2
```

### 单数据集重跑（跳过拆帧和预处理）

```bash
python pipelines/pipeline_runner.py \
  --data_root /path/to/output_root \
  --datasets your_dataset_name \
  --skip_frames \
  --skip_preprocess \
  --gpus 0
```

## 4. 关键参数

- `--datasets a,b,c`：仅处理指定数据集
- `--skip_frames` / `--skip_preprocess` / `--skip_colmap` / `--skip_2dgs` / `--skip_texture`
- `--use_colmap_mask 0|1`：COLMAP 特征提取是否使用 `mask`
- `--mesh_res`：2DGS 渲染导出网格分辨率
- `--num_view`：纹理阶段选帧数量

## 5. 注意事项

- 目录命名建议统一使用 `images` 与 `mask`。
- `step_2dgs.py` 会自动选择最新 `recon/train/ours_*` 并导出 `2dgs_recon.obj` 与 `transforms.json`。
- `colmap/` 为第三方目录，建议尽量不改动其内部内容。
