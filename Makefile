include ./colors.mk

.ONESHELL:
SHELL = /bin/bash
PYTHON = python3.12
VENV = lc_venv
BUILD_DIR = src playground

# Python Virtual Environment
.PHONY: venv
venv:
	@echo -e "$(COLOR_GREEN)Creating virtual environment...$(END_COLOR)"
	${PYTHON} -m venv ${VENV}
	@echo -e "$(COLOR_GREEN)Activating virtual environment and installing dependencies...$(END_COLOR)"
	source ${VENV}/bin/activate && \
	${PYTHON} -m pip install --default-timeout=1000 --upgrade pip && \
	${PYTHON} -m pip install --default-timeout=1000 -r requirements.txt

.PHONY: style
style:
	@echo -e "$(COLOR_GREEN)Running code style checks...$(END_COLOR)"
	source ${VENV}/bin/activate && \
	black ${BUILD_DIR} ; \
	isort ${BUILD_DIR} ; \
	flake8 ${BUILD_DIR}
