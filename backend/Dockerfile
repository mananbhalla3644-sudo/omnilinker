# syntax=docker/dockerfile:1
#
# OmniLinker backend image.
#
# Two stages, and the split is about a real concern rather than image size: the
# runtime stage carries no compiler and no package cache, so the attack surface
# of the process that holds the workspace keys is as small as it can be. The
# keystore file is the crown jewel - it is the only thing standing between a
# database dump and every message body in it.
FROM python:3.14-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
# --only-binary is not optional: pydantic-core has no source build path that
# works without a Rust toolchain in the image, and the failure is cryptic.
COPY requirements.txt .
RUN python -m pip install --only-binary :all: -r requirements.txt

FROM python:3.14-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMNI_DATA_DIR=/data \
    OMNI_MODE=docker

# curl for the healthcheck; nothing else.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 omni
WORKDIR /app

COPY --from=build /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY omnilinker ./omnilinker

# /data holds the keystore and, in embedded mode, the store. Owned by the
# unprivileged user so a bind mount inherits sane permissions.
RUN mkdir -p /data && chown -R omni:omni /data /app
USER omni

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/system/health || exit 1

# One worker on purpose: the embedded store and the in-memory search index are
# per-process. Scaling out means OMNI_MODE=docker with a real Mongo and Neo4j,
# where the stores are shared and the index can be rebuilt per worker.
CMD ["uvicorn", "omnilinker.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
