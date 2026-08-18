# How to update hashed requirement locks.
# See docs/dependencies.md.
#
#   uv pip compile pyproject.toml --python 3.12 --generate-hashes -o requirements/runtime.lock.txt
#   uv pip compile pyproject.toml --python 3.12 --extra dev --extra security --generate-hashes -o requirements/dev.lock.txt
