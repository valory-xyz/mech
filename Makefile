.PHONY: clean
clean: clean-build clean-pyc clean-test clean-docs

.PHONY: clean-build
clean-build:
	rm -fr deployments/build
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	rm -fr deployments/build/
	rm -fr deployments/Dockerfiles/open_aea/packages
	rm -fr pip-wheel-metadata
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +
	find . -name '*.svn' -exec rm -fr {} +
	rm -fr .idea .history
	rm -fr venv

.PHONY: clean-docs
clean-docs:
	rm -fr site/

.PHONY: clean-pyc
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '.DS_Store' -exec rm -fr {} +

.PHONY: clean-test
clean-test:
	rm -fr .tox/
	rm -f .coverage
	find . -name ".coverage*" -not -name ".coveragerc" -exec rm -fr "{}" \;
	rm -fr coverage.xml
	rm -fr htmlcov/
	rm -fr .hypothesis
	rm -fr .pytest_cache
	rm -fr .mypy_cache/
	find . -name 'log.txt' -exec rm -fr {} +
	find . -name 'log.*.txt' -exec rm -fr {} +

format: clean 
	poetry run isort packages scripts tests && \
	poetry run black packages scripts tests

is_dirty: clean
	if [ ! -z "$(shell git status -s)" ];\
	then\
		git diff;\
		echo "Branch is dirty exit";\
		exit 1;\
	fi;\

are_deps_dirty: clean
	git checkout master pyproject.toml
	if [ ! -z "$(shell git status -s pyproject.toml)" ];\
	then\
		git diff pyproject.toml;\
		echo "Dependencies are dirty exit!";\
		exit 1;\
	fi;\

lock:
	autonomy hash all && autonomy packages lock

run-single-agent:
	bash scripts/start_agent.sh

run-mas:
	bash scripts/start_multi_agent.sh
