# Build only after the running benchmark finishes. No GPU access is needed to build.
ARG BASE_IMAGE=tensorfold-dsv41:07aa02585f64
FROM ${BASE_IMAGE}
COPY apply.py /opt/display-kv-stage/apply.py
COPY overlay /opt/display-kv-stage/overlay
COPY probe.py /opt/display-kv-stage/probe.py
RUN python3 /opt/display-kv-stage/apply.py /opt/tensorfold
