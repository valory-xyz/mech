ARG AUTONOMY_IMAGE_VERSION="latest"
ARG AUTONOMY_IMAGE_NAME="valory/open-autonomy"
FROM ${AUTONOMY_IMAGE_NAME}:${AUTONOMY_IMAGE_VERSION}

# keep our wrapper first in PATH so we shadow any pipx shims
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH="/usr/local/bin:${PATH}"

# remove any pipx aea and old console scripts, then pin open-aea 1.65.0 and add a wrapper
RUN bash -lc 'command -v pipx >/dev/null && pipx uninstall aea || true' && \
    rm -f /root/.local/bin/aea || true && \
    python -m pip uninstall -y open-aea || true && \
    python -m pip install --no-cache-dir "open-aea==1.65.0" && \
    printf "#!/usr/bin/env bash\nexec python -m aea.cli \"\$@\"\n" > /usr/local/bin/aea && \
    chmod +x /usr/local/bin/aea && \
    which aea && aea --version && python -m aea.cli --version

ARG AEA_AGENT
ARG AUTHOR
ARG EXTRA_DEPENDENCIES

RUN aea init --reset --remote --ipfs --author ${AUTHOR}

WORKDIR /root

# This now runs with AEA 1.65.0 (wrapper is first on PATH)
RUN AEA_AGENT=${AEA_AGENT} EXTRA_DEPENDENCIES=${EXTRA_DEPENDENCIES} bash /root/scripts/install.sh

# OpenCV bits (unchanged)
RUN pip uninstall -y opencv-python || echo "opencv-python not installed"
RUN pip uninstall -y opencv-python-headless || echo "opencv-python-headless not installed"
RUN pip install --no-cache-dir opencv-python-headless==4.10.0.84

CMD ["/root/scripts/start.sh"]

HEALTHCHECK --interval=3s --timeout=600s --retries=600 \
  CMD netstat -ltn | grep -c 26658 > /dev/null || exit 1
