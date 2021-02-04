MODULES=src test

all: test

lint:
	flake8 $(MODULES)

check_readme:
	python setup.py check -r -s

tests:=$(wildcard test/test_*.py)

# A pattern rule that runs a single test module, for example:
#   make tests/test_gen3_input_json.py

$(tests): %.py : mypy lint check_readme
	python -m unittest --verbose $*.py

test: $(tests)

develop:
	pip install -e .
	pip install -Ur --upgrade-strategy eager requirements.dev.txt

undevelop:
	python setup.py develop --uninstall
	pip uninstall -y -r requirements.dev.txt

release: test
	python release.py $(version)

.PHONY: all lint mypy test develop undevelop release