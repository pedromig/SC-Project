SHELL := /bin/bash
ENV := venv
PYTHON := python3

SIMCX := src/simcx
SITE  := venv/lib/python3.10/site-packages
REQUIREMENTS := requirements.txt

.PHONY: env

env:
	@echo "Setting up virtual environment..."
	@$(PYTHON) -m venv $(ENV)
	@source $(ENV)/bin/activate && \
		$(PYTHON) -m pip install --upgrade pip && \
		$(PYTHON) -m pip install -r $(REQUIREMENTS)
	@cp -r $(SIMCX) $(SITE)	
	@echo "DONE!" 

clean:
	@echo -n "Removing environment... "
	@rm -rf $(ENV)
	@echo "DONE!"
	
