SRC   := func_llm
TESTS := tests

release:
	git tag v$$(uv version --short)
	git push origin v$$(uv version --short)

ci-dependencies:
	pip install uv
	uv sync --no-install-project --all-extras

ci-test:
	uv run ruff format --check
	uv run ruff check
	uv run mypy --strict $(SRC) $(TESTS)
	uv run pytest $(TESTS)

ci-build:
	uv build

ci-publish:
	uv publish --index "va-pypi"
