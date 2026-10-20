# Build the ordinary pinned image first. This image is independent of display KV.
ARG BASE_IMAGE=tensorfold-dsv41:07aa02585f64
FROM ${BASE_IMAGE}
COPY experiments/memory-topk /opt/memory-topk
RUN python /opt/memory-topk/apply.py /opt/tensorfold
# Do not enable TF_DS_MEM_CAP here. Selecting this image enables only bounded top-k.
