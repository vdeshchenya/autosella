# syntax=docker/dockerfile:1
# Required build arguments: no tool-version defaults or host account state.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BUN_VERSION
ARG RALPH_COMMIT
ARG CODEX_VERSION
ARG CLAUDE_VERSION
LABEL org.gigaopt.isolated-runs="1"
RUN apt-get update && apt-get install --no-install-recommends -y \
      ca-certificates curl git openssh-client python3 python-is-python3 python3-venv iptables util-linux unzip \
    && rm -rf /var/lib/apt/lists/*
RUN npm install --global "@openai/codex@${CODEX_VERSION}" "@anthropic-ai/claude-code@${CLAUDE_VERSION}" \
    && npm cache clean --force
RUN curl --fail --silent --show-error --location https://bun.sh/install -o /tmp/install-bun.sh \
    && BUN_INSTALL=/opt/bun bash /tmp/install-bun.sh "bun-v${BUN_VERSION}" \
    && ln -s /opt/bun/bin/bun /usr/local/bin/bun && rm /tmp/install-bun.sh
RUN git init /opt/ralph && cd /opt/ralph \
    && git remote add origin https://github.com/Th0rgal/open-ralph-wiggum.git \
    && git fetch --depth=1 origin "${RALPH_COMMIT}" && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${RALPH_COMMIT}" \
    && bun install --frozen-lockfile && rm -rf .git
COPY isolated_runs/patch_ralph.py /opt/autoresearch/isolated_runs/patch_ralph.py
RUN python3 /opt/autoresearch/isolated_runs/patch_ralph.py /opt/ralph/ralph.ts
RUN printf '#!/bin/sh\nexec /usr/local/bin/bun /opt/ralph/ralph.ts "$@"\n' > /usr/local/bin/ralph \
    && chmod 755 /usr/local/bin/ralph
COPY isolated_runs /opt/autoresearch/isolated_runs
COPY evaluation_access /opt/autoresearch/evaluation_access
RUN codex --version && claude --version && ralph --help >/dev/null \
    && python3 -m compileall -q /opt/autoresearch \
    && mkdir -p /workspace /home/research /scratch /memory /tools /dataset
ENV PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python3", "/opt/autoresearch/isolated_runs/runner.py"]
