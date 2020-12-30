.PHONY: all lint

all_tests: lint unittest integration

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  lint			to run flake8 on all Python files"
	@echo "  unittest		to run unit tests on phys2denoise"
	@echo "  integration	to run the integration test set on phys2denoise"
	@echo "  all_tests		to run 'lint', 'unittest', and 'integration'"

lint:
	@flake8 phys2denoise

unittest:
	@py.test --skipintegration --cov-append --cov-report term-missing --cov=phys2denoise phys2denoise/

integration:
	@pip install -e ".[test]"
	@py.test --log-cli-level=INFO --cov-append --cov-report term-missing --cov=phys2denoise -k test_integration phys2denoise/tests/test_integration.py
