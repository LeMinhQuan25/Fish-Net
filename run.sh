#!/usr/bin/env bash
set -e
python -m pip install -e .
python tests/sanity_check.py
