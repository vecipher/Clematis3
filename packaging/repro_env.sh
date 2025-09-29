#!/usr/bin/env bash
set -euo pipefail
# Repro knobs
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1700000000}"  # 2023-11-14T22:13:20Z
export TZ=UTC
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONHASHSEED=0
export PYTHONUTF8=1
export UMASK=0022
umask 0022

# pip/build noise off; network stays off in CI by policy
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_PYTHON_VERSION_WARNING=1
