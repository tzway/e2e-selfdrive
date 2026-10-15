#!/usr/bin/env bash
# 按运行平台自动安装依赖:
#   x86_64 + NVIDIA GPU    -> CUDA 版(torch cu128 / tensorflow and-cuda)
#   aarch64(Apple Silicon) -> 原生 arm64 CPU 版
#   x86_64 无 GPU          -> CPU 版
set -euo pipefail

# 网络不稳时自动重试(大文件下载中途被掐断、校验失败时很有用)
retry() {
  local n=0
  until "$@"; do
    n=$((n + 1))
    if [ "$n" -ge 3 ]; then
      echo ">>> 重试 $n 次仍失败: $*" >&2
      return 1
    fi
    echo ">>> 失败,自动重试 ($n/3): $*" >&2
    sleep 3
  done
}

# 1) 系统依赖(Debian 12 里 libGL 的包名是 libgl1,不是旧系统的 libgl1-mesa-glx)
# 旧版 devcontainers 镜像内置的 yarn 源签名密钥已失效,会让 apt-get update 报错退出;
# 本仓库用不到该 apt 源,先禁用它
if [ -f /etc/apt/sources.list.d/yarn.list ]; then
  sudo mv /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn.list.disabled
fi
retry sudo apt-get update
retry sudo apt-get install -y libgl1 libglib2.0-0

# 2) 各平台公共依赖
# (opencv 固定 4.12 稳定线;imgaug 也依赖这两个包,版本保持一致避免互相覆盖)
common_pkgs=(
  opencv-python==4.12.0.88 opencv-python-headless==4.12.0.88
  numpy matplotlib keras
  python-engineio==3.13.2 python-socketio==4.6.1
  Flask eventlet scikit-learn notebook pandas
)

ARCH="$(uname -m)"
if command -v nvidia-smi >/dev/null 2>&1 || [ -e /dev/nvidia0 ]; then
  HAS_GPU=true
else
  HAS_GPU=false
fi

if [ "$ARCH" = "x86_64" ] && [ "$HAS_GPU" = true ]; then
  echo ">>> 平台: x86_64 + NVIDIA GPU,安装 CUDA 版依赖"
  retry pip install "${common_pkgs[@]}" "tensorflow[and-cuda]"
  retry pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
elif [ "$ARCH" = "aarch64" ]; then
  echo ">>> 平台: aarch64(Apple Silicon),安装 arm64 CPU 版依赖"
  retry pip install "${common_pkgs[@]}" tensorflow
  retry pip install torch torchvision
else
  echo ">>> 平台: x86_64(无 GPU),安装 CPU 版依赖"
  retry pip install "${common_pkgs[@]}" tensorflow
  retry pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# 3) 项目专用:imgaug 数据增强(新版只发布在 git 仓库上)
# 优先用 codeload tarball 直装(纯 HTTPS 下载,对不稳定网络更稳),git clone 作兜底
if ! retry pip install "https://github.com/imaug/imaug/archive/refs/heads/master.tar.gz"; then
  retry pip install git+https://github.com/imaug/imaug.git
fi

echo ">>> 依赖安装完成"
