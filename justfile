KAAJ_HAALKA_COMMIT := "27104c4f47980cb53ec64c6d5e7874cf4f3caaec"

fetch_kaaj_justfile:
  curl https://raw.githubusercontent.com/databasedav/haalka/{{ KAAJ_HAALKA_COMMIT }}/kaaj/justfile > kaaj.just

import? 'kaaj.just'

# TODO: use an actual list https://github.com/casey/just/issues/2458
exclude_examples := '"test", "utils"'

# TODO: use an actual list https://github.com/casey/just/issues/2458
export_nickels := "ci build_example pr_previews examples_on_main cleanup_pr_previews release"

repo_prompt:
  @nickel eval repo_prompt.ncl | sed 's/^"//; s/"$//; s/\\"/"/g; s/\\n/\n/g'

sync_readme_example:
  uv run python sync_readme_example.py
