set allow-duplicate-recipes

KAAJ_HAALKA_COMMIT := "7064230a1c6f7f1b562301ed3d51cf2f8f8a9a01"

fetch_kaaj_justfile:
  curl https://raw.githubusercontent.com/databasedav/haalka/{{ KAAJ_HAALKA_COMMIT }}/kaaj/justfile > kaaj.just

import? 'kaaj.just'

test:
  cargo test tests -- --test-threads=1 && just doctest

# TODO: use an actual list https://github.com/casey/just/issues/2458
exclude_examples := '"test", "utils"'

# TODO: use an actual list https://github.com/casey/just/issues/2458
export_nickels := "ci build_example pr_previews examples_on_main cleanup_pr_previews release"

repo_prompt:
  @nickel eval repo_prompt.ncl | sed 's/^"//; s/"$//; s/\\"/"/g; s/\\n/\n/g'

sync_readme_example:
  uv run python sync_readme_example.py
