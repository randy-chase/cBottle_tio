# SPDX-License-Identifier: Apache-2.0
lint:
	python3 ci/check_licenses.py
	python -m ruff format --check
	python -m ruff check

docs:
	uvx mkdocs build

pytest:
	torchrun --nproc_per_node 1 -m pytest -vv -rs

.PHONY: docs
