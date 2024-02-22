.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = xray_classifier
PYTHON_INTERPRETER = python3
VERSION=0.1
VENV=${PROJECT_NAME}-${VERSION}
JUPYTER_ENV_NAME=${VENV}
JUPYTER_PORT=8888
PYENV = ${PYENV_ROOT}


ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif



#################################################################################
# COMMANDS                                                                      #
#################################################################################

init:
	git init
	git add -A
	git commit -m "Initial commit"
	git branch -M main
	gh repo create
	@echo "push with: git push -u origin main"

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Set up python interpreter environment
create_environment:
	eval "$(pyenv init --path)"
	@echo "--> Install and setup pyenv and virtualenv"
	python3 -m pip install --upgrade pip
## ${PYENV}/bin/pyenv -p ${PYTHON_INTERPRETER} ${VENV}
## pyenv -p ${PYTHON_INTERPRETER} ${VENV}
##	echo ${VENV} > python-version

ipykernel: create_environment ##@main >> install a Jupyter iPython kernel using our virtual environment
	@echo ""
	@echo "--> Install ipykernel to be used by jupyter notebooks"
	$(PYTHON_INTERPRETER) -m pip install ipykernel jupyter jupyter_contrib_nbextensions
	$(PYTHON_INTERPRETER) -m ipykernel install --user --name=$(VENV) --display-name=$(JUPYTER_ENV_NAME)
	$(PYTHON_INTERPRETER) -m jupyter nbextension enable --py widgetsnbextension --sys-prefix

jupyter: create_environment ##@main >> start a jupyter notebook
	@echo ""
	@echo "--> Running jupyter notebook on port $(JUPYTER_PORT)"
	jupyter notebook --port $(JUPYTER_PORT)
## Test python environment is setup correctly
test_environment:
##	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
