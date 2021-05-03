#!/usr/bin/env bash

set -o errexit
set -o nounset

if [[ ! -d venv ]]; then
    if [[ -e venv ]]; then
        echo "E: venv exists but is not a directory!" >&2
        exit 1
    fi

    echo "Virtual environment venv does not exist. Creating..."
    if which virtualenv; then
        virtualenv venv
    else
        echo "E: virtualenv has to be installed!" >&2
        exit 2
    fi

    echo "Activating virtual environment venv..."
    source venv/bin/activate
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "Activating virtual environment venv..."
    source venv/bin/activate
fi

PYTHONPATH=. python src/main.py with ex_config_example
