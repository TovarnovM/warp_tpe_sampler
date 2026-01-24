# syntax=docker/dockerfile:1
FROM python:3.12-bookworm

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      curl \
      ca-certificates \
      bash \
      sudo \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user. Be robust if USER_UID/USER_GID are empty.
RUN set -eux; \
    UID_TO_USE="${USER_UID:-1000}"; \
    GID_TO_USE="${USER_GID:-1000}"; \
    if [ -z "${UID_TO_USE}" ]; then UID_TO_USE="1000"; fi; \
    if [ -z "${GID_TO_USE}" ]; then GID_TO_USE="1000"; fi; \
    if ! getent group "${USERNAME}" >/dev/null; then \
      groupadd -g "${GID_TO_USE}" "${USERNAME}" || groupadd "${USERNAME}"; \
    fi; \
    if ! id -u "${USERNAME}" >/dev/null 2>&1; then \
      useradd -m -s /bin/bash -u "${UID_TO_USE}" -g "${USERNAME}" "${USERNAME}" || \
      useradd -m -s /bin/bash -g "${USERNAME}" "${USERNAME}"; \
    fi; \
    usermod -aG sudo "${USERNAME}"; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"

WORKDIR /workspaces/warp_tpe_sampler
USER ${USERNAME}
SHELL ["/bin/bash", "-lc"]
