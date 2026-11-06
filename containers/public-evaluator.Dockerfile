# Public source and public dataset cache only. GPU serving and CPU evaluation use separate environments.
FROM vllm/vllm-openai@sha256:c6e88abc61f250e3b0d72489b077107e92fd004c1270577700ef85dd3f41755d
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/eval
COPY delta_cli/public_eval/requirements.txt /tmp/public-evaluator-requirements.txt
RUN /opt/eval/bin/pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu && /opt/eval/bin/pip install --no-cache-dir -r /tmp/public-evaluator-requirements.txt
# Explicit allowlist: research records, workspace files and local credentials never enter the image.
COPY pyproject.toml MANIFEST.in README.md ARTIFACTS.md LOCAL.md PUBLIC_EVALUATION.md /app/
COPY delta_cli /app/delta_cli
RUN /opt/eval/bin/pip install --no-cache-dir /app
ENV HF_HOME=/opt/hf NLTK_DATA=/opt/nltk_data
RUN /opt/eval/bin/python -m delta_cli.public_eval.cache
RUN /opt/eval/bin/python -m delta_cli.public_eval.install_wasi /opt/python-wasi
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
ENTRYPOINT ["/opt/eval/bin/delta", "public-eval"]
CMD ["--help"]
