#!/bin/bash

sudo apt install -y graphviz libasound2-dev

curl -sSL https://install.python-poetry.org | python3 -
poetry install
poetry run maturin develop
