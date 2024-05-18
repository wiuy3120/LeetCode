# .ONESHELL:
SHELL = /bin/bash
PYTHON = python3.8
VENV = lc_venv
BUILD_DIR = src playground

# Python Virtual Enviroment
.PHONY: venv
venv:
	${PYTHON} -m venv ${VENV}
	source ${VENV}/bin/activate && \
	${PYTHON} -m pip install --default-timeout=1000 --upgrade pip && \
	${PYTHON} -m pip install --default-timeout=1000 -r requirements.txt

.PHONY: style
style:
	$(ENV_PREFIX)black ${BUILD_DIR}
	$(ENV_PREFIX)isort ${BUILD_DIR}
	$(ENV_PREFIX)flake8 ${BUILD_DIR}
	
	