# Makefile


.PHONY: clean

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -

.PHONY: lint
lint:
	pylint src

.PHONY: format
format:
	black src
