# TensorFold (Python engine, DeepSeek-V4.1 family) on NVIDIA's NGC PyTorch 26.07 base for DGX Spark (GB10, sm_121,
# aarch64, CUDA 13). The "grammar" extra adds xgrammar (structured output). The constraint file holds NGC's own torch
# build, so pip does not replace it with a PyPI wheel. Build with docker/build.sh (it passes TF_REPO and TF_COMMIT).
FROM nvcr.io/nvidia/pytorch:26.07-py3

ARG TF_REPO=https://github.com/Elevatormusic/TensorFold.git
ARG TF_COMMIT
RUN test -n "$TF_COMMIT" || { echo "set --build-arg TF_COMMIT=<commit>"; exit 1; } \
 && git clone --filter=blob:none "$TF_REPO" /opt/tensorfold \
 && git -C /opt/tensorfold checkout --detach "$TF_COMMIT"
RUN pip freeze | grep -iE "^(torch|torchvision|triton|numpy)==" > /tmp/ngc-pins.txt \
 && pip install --no-cache-dir -c /tmp/ngc-pins.txt -e "/opt/tensorfold[grammar]" \
 && python -c "import importlib.metadata, tensorfold, torch; print('tensorfold', tensorfold.__file__, 'xgrammar', importlib.metadata.version('xgrammar'), 'torch', torch.__version__)"
ENV TORCH_EXTENSIONS_DIR=/cache/torch_extensions
WORKDIR /opt/tensorfold
