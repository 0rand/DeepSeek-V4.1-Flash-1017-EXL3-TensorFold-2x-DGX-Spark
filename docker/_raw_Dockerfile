# TensorFold DeepSeek-V4.1 Python engine (Elevatormusic/TensorFold, branch v41-dsml-tool-choice) on the NGC
# PyTorch 26.07 base. The grammar extra adds xgrammar (structured output). The constraint file holds the NGC
# torch build, so pip does not replace it with a PyPI wheel.
FROM nvcr.io/nvidia/pytorch:26.07-py3
COPY src /opt/tensorfold
RUN pip freeze | grep -iE "^(torch|torchvision|triton|numpy)==" > /tmp/ngc-pins.txt \
 && pip install --no-cache-dir -c /tmp/ngc-pins.txt -e "/opt/tensorfold[grammar]" \
 && python -c "import importlib.metadata, tensorfold, xgrammar, torch; print('tensorfold', tensorfold.__file__, 'xgrammar', importlib.metadata.version('xgrammar'), 'torch', torch.__version__, torch.cuda.is_available())"
ENV TORCH_EXTENSIONS_DIR=/cache/torch_extensions
WORKDIR /opt/tensorfold
