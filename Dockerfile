FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GSPLAT_TAG=v1.5.3
ARG PROPAINTER_TAG=v0.1.0

ENV TZ=Asia/Shanghai \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=8.6 \
    MAX_JOBS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      libsm6 \
      libxext6 \
      libxrender1 \
      ninja-build \
      python3.10 \
      python3.10-dev \
      python3-pip \
      python3-venv \
      wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel

# Driver 530 cannot run the cu128 environment recorded in the SOP. PyTorch
# 2.7.1 is pinned to the official cu118 wheel set instead.
RUN python -m pip install \
      torch==2.7.1 \
      torchvision==0.22.1 \
      torchaudio==2.7.1 \
      --index-url https://download.pytorch.org/whl/cu118

RUN python -m pip install \
      "numpy<2.0" \
      pycolmap==4.1.1 \
      ultralytics==8.4.172 \
      scipy \
      opencv-python \
      Pillow \
      tqdm \
      scikit-learn \
      "torchmetrics[image]" \
      tyro==1.0.16 \
      tensorboard \
      tensorly \
      pyyaml \
      matplotlib \
      splines \
      imageio \
      imageio-ffmpeg \
      av \
      addict \
      einops \
      future \
      scikit-image \
      requests \
      timm \
      yapf

COPY patches/pycolmap_scene_manager_compat.py \
     /tmp/pycolmap_scene_manager_compat.py

# Pin the upstream trainer code and all git dependencies used by its example.
RUN git clone --branch "${GSPLAT_TAG}" --depth 1 \
      --recurse-submodules --shallow-submodules \
      https://github.com/nerfstudio-project/gsplat.git /root/gsplat \
    && test -f /root/gsplat/gsplat/cuda/csrc/third_party/glm/glm/gtc/type_ptr.hpp \
    && cp /tmp/pycolmap_scene_manager_compat.py \
      /root/gsplat/examples/pycolmap_scene_manager_compat.py \
    && sed -i \
      's/from pycolmap import SceneManager/from pycolmap_scene_manager_compat import SceneManager/' \
      /root/gsplat/examples/datasets/colmap.py \
    && grep -q \
      'from pycolmap_scene_manager_compat import SceneManager' \
      /root/gsplat/examples/datasets/colmap.py \
    && python -m pip install --no-build-isolation /root/gsplat \
    && python -m pip install --no-build-isolation \
      "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787" \
      "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5" \
      "git+https://github.com/harry7557558/fused-bilagrid@90f9788e57d3545e3a033c1038bb9986549632fe"

# ProPainter source and the exact three weights required by the SOP.
RUN git clone --branch "${PROPAINTER_TAG}" --depth 1 \
      https://github.com/sczhou/ProPainter.git /opt/ProPainter \
    && mkdir -p /opt/ProPainter/weights \
    && wget -q --tries=10 --timeout=30 \
      https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth \
      -O /opt/ProPainter/weights/ProPainter.pth \
    && wget -q --tries=10 --timeout=30 \
      https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth \
      -O /opt/ProPainter/weights/raft-things.pth \
    && wget -q --tries=10 --timeout=30 \
      https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth \
      -O /opt/ProPainter/weights/recurrent_flow_completion.pth \
    && test "$(stat -c %s /opt/ProPainter/weights/ProPainter.pth)" = 157780510 \
    && test "$(stat -c %s /opt/ProPainter/weights/raft-things.pth)" = 21108000 \
    && test "$(stat -c %s /opt/ProPainter/weights/recurrent_flow_completion.pth)" = 20348681

# Download the model during the connected build, never on the offline server.
RUN mkdir -p /opt/c4/models \
    && cd /opt/c4/models \
    && python -c "from ultralytics import YOLO; YOLO('yolo11x-seg.pt')" \
    && test -s /opt/c4/models/yolo11x-seg.pt

RUN python - <<'PY'
import pycolmap
import torch
import torchvision
import gsplat
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("pycolmap", pycolmap.__version__)
print("gsplat", gsplat.__file__)
assert torch.__version__.startswith("2.7.1")
assert torch.version.cuda == "11.8"
assert pycolmap.__version__ == "4.1.1"
PY

RUN cd /root/gsplat/examples \
    && python simple_trainer.py default --help > /tmp/gsplat-help.txt \
    && grep -q -- '--near-plane' /tmp/gsplat-help.txt \
    && grep -q -- '--strategy.refine-stop-iter' /tmp/gsplat-help.txt \
    && grep -q -- '--strategy.reset-every' /tmp/gsplat-help.txt \
    && grep -q -- '--save-ply' /tmp/gsplat-help.txt \
    && grep -q -- '--ply-steps' /tmp/gsplat-help.txt

RUN git -C /root/gsplat rev-parse HEAD > /opt/c4/gsplat.commit \
    && git -C /opt/ProPainter rev-parse HEAD > /opt/c4/propainter.commit \
    && python -m pip freeze > /opt/c4/python-freeze.txt \
    && sha256sum \
      /opt/c4/models/yolo11x-seg.pt \
      /opt/ProPainter/weights/ProPainter.pth \
      /opt/ProPainter/weights/raft-things.pth \
      /opt/ProPainter/weights/recurrent_flow_completion.pth \
      > /opt/c4/model-SHA256SUMS.txt

WORKDIR /root/gsplat/examples

CMD ["bash"]
