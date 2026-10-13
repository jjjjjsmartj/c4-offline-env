# C4 离线复现环境

这个目录用于在 GitHub Actions 上构建完整 Docker 环境，再通过 GitHub Release 分卷下载和 U 盘传输到离线 GPU 服务器。

## 固定版本

- 基础镜像 `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- Python 3.10
- PyTorch 2.7.1 CUDA 11.8
- torchvision 0.22.1
- pycolmap 4.1.1 CPU wheel
- gsplat v1.5.3
- Ultralytics 8.4.172
- ProPainter v0.1.0 代码和三份官方权重

gsplat、fused-ssim 和 fused-bilagrid 使用 `--no-build-isolation` 编译，因为它们的构建脚本需要直接导入已经安装在镜像环境中的 PyTorch。
gsplat 使用 `--recurse-submodules --shallow-submodules` 克隆，以包含 CUDA 扩展编译所需的 GLM 头文件。
gsplat v1.5.3 示例所需的旧 `SceneManager` 接口由 `patches/pycolmap_scene_manager_compat.py` 适配到官方 pycolmap 4.1.1；不会降级或覆盖 SOP 使用的 pycolmap。

SOP 记录的 PyTorch 2.9.1 CUDA 12.8 不适用于驱动 530.41.03，因此这里使用仍由 PyTorch 官方提供 CUDA 11.8 wheel 的 2.7.1。

## 上传 GitHub

新建一个公开仓库，把本目录中的全部内容上传到仓库根目录。最终仓库根目录应直接包含：

```text
Dockerfile
README.md
.github/workflows/build-offline-c4-env.yml
```

进入 Actions，手动运行 `Build offline C4 reconstruction environment`。成功后，从 Release `c4-repro-cu118-v1` 下载全部 `part-*` 文件和 `SHA256SUMS.txt`。

## 离线导入

把全部文件放入 GPU 服务器 `/devdata/env/c4-repro-cu118-v1`：

```bash
cd /devdata/env/c4-repro-cu118-v1
sha256sum -c SHA256SUMS.txt
cat c4-repro-cu118-v1.tar.gz.part-* | gzip -dc | docker load
```

## GPU 验证

```bash
docker run --rm \
  --gpus '"device=0"' \
  c4-repro:cu118-v1 \
  bash -lc '
    python - <<"PY"
import torch, torchvision, pycolmap, gsplat
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("torchvision:", torchvision.__version__)
print("pycolmap:", pycolmap.__version__)
print("gsplat:", gsplat.__file__)
PY
  '
```

预期至少包含：

```text
torch: 2.7.1+cu118
CUDA runtime: 11.8
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
pycolmap: 4.1.1
```

## 后续运行容器

将数据和脚本放在 GPU 服务器 `/devdata/c4_repro`，然后创建持久容器：

```bash
docker run -it \
  --name c4-repro \
  --gpus '"device=0"' \
  --shm-size=16g \
  -v /devdata/c4_repro:/root/autodl-tmp/codex \
  c4-repro:cu118-v1 \
  bash
```

容器退出后重新进入：

```bash
docker start c4-repro
docker exec -it c4-repro bash
```

镜像内资源位置：

```text
/root/gsplat
/opt/ProPainter
/opt/ProPainter/weights
/opt/c4/models/yolo11x-seg.pt
/opt/c4/python-freeze.txt
/opt/c4/model-SHA256SUMS.txt
```

首次进入后，将在线构建阶段下载好的 YOLO 权重复制到数据目录：

```bash
mkdir -p /root/autodl-tmp/codex/c4_source/models
cp -n /opt/c4/models/yolo11x-seg.pt \
  /root/autodl-tmp/codex/c4_source/models/
```

SOP 中 ProPainter 的路径应从 `/root/autodl-tmp/codex/ProPainter` 改为 `/opt/ProPainter`。统一环境已经包含 ProPainter 依赖，不再创建或激活 `venvs/dynmask`。
