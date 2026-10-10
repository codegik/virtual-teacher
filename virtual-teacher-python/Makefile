# The one way to run this project. Don't call uv/python/pytest by hand.
SHELL := /bin/bash
export PATH := $(HOME)/.local/bin:$(PATH)
# CUDA libraries come from the venv's nvidia-* wheels (no system CUDA, no sudo).
NV := $(CURDIR)/.venv/lib/python3.12/site-packages/nvidia
export PYTHONPATH := $(CURDIR)/src
export LD_LIBRARY_PATH := $(NV)/cuda_runtime/lib:$(NV)/cublas/lib:$(NV)/cudnn/lib:$(LD_LIBRARY_PATH)

.PHONY: help setup doctor run test bench probe

help:
	@echo "make setup   install Python packages and download the models. Safe to re-run"
	@echo "make doctor  check that everything is installed, without changing anything"
	@echo "make run     open Pipo (one process: window, microphone, Whisper, qwen2.5, Kokoro)"
	@echo "make test    full test suite"
	@echo "make bench   time a real turn with the real models (time to first audio)"
	@echo "make probe   measure the room (music, singing) vs someone talking to Pipo, for the barge-in bar"

setup:
	@scripts/setup.sh

doctor:
	@scripts/setup.sh --check

run: setup
	.venv/bin/python -m pipo

test: doctor
	.venv/bin/pytest

bench: doctor
	.venv/bin/python -m pipo.bench

probe: doctor
	.venv/bin/python -m pipo.probe
