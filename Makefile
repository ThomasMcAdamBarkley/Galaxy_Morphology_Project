# Galaxy Morphology Project — task runner
# Requires GNU Make (available via Git Bash / WSL on Windows)
#
# Usage:
#   make install                          install dependencies
#   make train                            train the 3-class augmented model
#   make train-hierarchical               train all 3 decision-tree nodes
#   make train-hierarchical NODE=edge     train a single node (top|edge|bar)
#   make evaluate                         classification report + confusion matrix
#   make predict IMAGE=data/images_train/100008.jpg
#   make gradcam  IMAGE=data/images_train/100008.jpg

.PHONY: install train train-all train-hierarchical train-bar evaluate predict gradcam clean

# ── defaults ──────────────────────────────────────────────────────────────────
IMAGE ?= data/images_train/100008.jpg
NODE  ?= all

PYTHON ?= .venv/bin/python

# ── setup ─────────────────────────────────────────────────────────────────────
venv:
	python3 -m venv .venv

install: venv
	.venv/bin/pip install --upgrade pip -q
	.venv/bin/pip install -r requirements.txt

# ── training ──────────────────────────────────────────────────────────────────
train:
	$(PYTHON) src/train_augmented.py

train-hierarchical:
	$(PYTHON) src/train_hierarchical.py --node $(NODE)

train-all:
	$(PYTHON) src/train_hierarchical.py --node all

train-bar:
	$(PYTHON) src/train_bar_efficientnet.py

# ── evaluation ────────────────────────────────────────────────────────────────
evaluate:
	$(PYTHON) src/evaluate.py

# ── inference ─────────────────────────────────────────────────────────────────
predict:
	$(PYTHON) src/predict.py --image $(IMAGE)

gradcam:
	$(PYTHON) src/analyze_native_reconstruct.py --image $(IMAGE)

# ── housekeeping ──────────────────────────────────────────────────────────────
clean:
	rm -rf outputs/
	@echo "outputs/ cleared."
