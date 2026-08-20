# predicators-sandbox — Docker image for running the Claude agent in isolation.
#
# Based on the robocode-sandbox pattern:
#   https://github.com/anthropics/claude-code/tree/main/.devcontainer
#
# Provides:
#   - Claude CLI (required by ClaudeSDKClient)
#   - Python 3.10 + all predicators dependencies
#   - claude-agent-sdk for in-process MCP tools
#   - Firewall restricting outbound to api.anthropic.com + GitHub
#
# Build from the repository root:
#   bash docker/build.sh
#
# Runtime bind-mounts (see docker_sandbox.py):
#   -v <repo>:/opt/predicators:ro   predicators source (for Python imports)
#   -v <sandbox_dir>:/sandbox       curated sandbox with reference files
#   -v <tmp_dir>:/data              shared data (pickled context + results)

FROM node:20

ARG TZ
ENV TZ="$TZ"

ARG CLAUDE_CODE_VERSION=latest

# ---------------------------------------------------------------------------
# System packages
#   - git, sudo, curl: basic dev tools
#   - jq, dnsutils (dig), aggregate, iptables, ipset, iproute2: firewall
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        sudo \
        curl \
        jq \
        iptables \
        ipset \
        iproute2 \
        dnsutils \
        aggregate \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Miniconda — provides Python 3.10 to match the host environment.
# The node:20 base ships Python 3.11 which is incompatible with dill
# pickles created by the host's Python 3.10 (CodeType arg count differs).
# ---------------------------------------------------------------------------
RUN ARCH=$(uname -m) \
    && curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh" \
           -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda install -y python=3.10 \
    && /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:$PATH"

# ---------------------------------------------------------------------------
# npm-global directory (mirrors official devcontainer setup)
# ---------------------------------------------------------------------------
RUN mkdir -p /usr/local/share/npm-global \
    && chown -R node:node /usr/local/share

ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH="$PATH:/usr/local/share/npm-global/bin"

# ---------------------------------------------------------------------------
# Claude Code CLI (required by ClaudeSDKClient)
# ---------------------------------------------------------------------------
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# ---------------------------------------------------------------------------
# Firewall script — node user gets passwordless sudo for this one script only
# ---------------------------------------------------------------------------
COPY docker/init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod +x /usr/local/bin/init-firewall.sh \
    && echo "node ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" \
       > /etc/sudoers.d/node-firewall \
    && chmod 0440 /etc/sudoers.d/node-firewall

# ---------------------------------------------------------------------------
# Claude config directory for the node user
# ---------------------------------------------------------------------------
RUN mkdir -p /home/node/.claude \
    && chown -R node:node /home/node/.claude

# ---------------------------------------------------------------------------
# Python environment — install predicators dependencies
#
# We copy setup.py and install deps only.  The actual source code is
# mounted read-only at /opt/predicators at runtime via bind-mount, so
# code changes never require a rebuild — only PyPI dependency changes do.
# ---------------------------------------------------------------------------
WORKDIR /build
COPY setup.py ./
RUN mkdir -p predicators && touch predicators/__init__.py \
    && pip install --no-cache-dir -e . || true
RUN pip install --no-cache-dir \
    claude-agent-sdk dill nest_asyncio "setuptools<81"

# ---------------------------------------------------------------------------
# Cleanup and create directories
# ---------------------------------------------------------------------------
RUN rm -rf /build \
    && mkdir -p /sandbox /data \
    && chmod 777 /sandbox /data

# ---------------------------------------------------------------------------
# Entrypoint: init firewall -> exec command
# ---------------------------------------------------------------------------
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV DEVCONTAINER=true
ENV CLAUDE_CONFIG_DIR=/home/node/.claude
ENV NODE_OPTIONS="--max-old-space-size=4096"

USER node
WORKDIR /sandbox

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
