FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    USE_TF=0 \
    USE_FLAX=0

COPY release-linux-amd64.lock /opt/domyn-release/requirements.lock
RUN python -m pip install --no-cache-dir --no-deps --only-binary=:all: \
        -r /opt/domyn-release/requirements.lock \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation \
        https://github.com/google-ai-edge/litert-torch/archive/13644e33d20cb18b48c7e5fbb2698fa5795740bf.tar.gz \
    && python -m pip check

WORKDIR /workspace
