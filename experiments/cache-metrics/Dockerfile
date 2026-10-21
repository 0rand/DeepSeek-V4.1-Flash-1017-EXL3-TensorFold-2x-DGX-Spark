FROM tensorfold-dsv41:07aa02585f64
COPY apply.py /opt/cache-metrics/apply.py
COPY overlay/cache_metrics.py /opt/cache-metrics/overlay/cache_metrics.py
RUN python /opt/cache-metrics/apply.py /opt/tensorfold
WORKDIR /opt/tensorfold
