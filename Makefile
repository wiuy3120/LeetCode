# .ONESHELL:
SHELL = /bin/bash
PYTHON = python3.12
VENV = lc_venv
BUILD_DIR = src playground

# Python Virtual Environment
.PHONY: venv
venv:
	${PYTHON} -m venv ${VENV}
	source ${VENV}/bin/activate && \
	${PYTHON} -m pip install --default-timeout=1000 --upgrade pip && \
	${PYTHON} -m pip install --default-timeout=1000 -r requirements.txt

.PHONY: style
style:
	source ${VENV}/bin/activate && \
	black ${BUILD_DIR} ; \
	isort ${BUILD_DIR} ; \
	flake8 ${BUILD_DIR}
