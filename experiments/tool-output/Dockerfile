# Build a new image. Never replace the ordinary image tag.
ARG BASE_IMAGE=tensorfold-dsv41:07aa02585f64
FROM ${BASE_IMAGE}
COPY apply.py /opt/tool-output/apply.py
COPY overlay /opt/tool-output/overlay
RUN python /opt/tool-output/apply.py /opt/tensorfold
