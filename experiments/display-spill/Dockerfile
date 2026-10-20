# Build from the repository root so both feature folders are in the context:
#   docker build -f experiments/display-spill/Dockerfile -t tensorfold-dsv41:display-spill .
ARG BASE_IMAGE=tensorfold-dsv41:07aa02585f64
FROM ${BASE_IMAGE}
COPY experiments/display-kv /opt/display-kv
COPY experiments/display-spill /opt/display-spill
RUN python /opt/display-spill/compose.py /opt/tensorfold /opt/display-kv/apply.py /opt/display-spill/apply.py
