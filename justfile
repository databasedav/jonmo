set allow-duplicate-recipes

KAAJ_HAALKA_COMMIT := "d44713752c766eab3ae72b66526934dc65322a57"

fetch_kaaj_justfile:
  curl https://raw.githubusercontent.com/databasedav/haalka/{{ KAAJ_HAALKA_COMMIT }}/kaaj/justfile > kaaj.just

import? 'kaaj.just'

test:
  cargo test tests -- --test-threads=1 && just doctest

# TODO: use an actual list https://github.com/casey/just/issues/2458
exclude_examples := '"test", "utils"'

# TODO: use an actual list https://github.com/casey/just/issues/2458
export_nickels := "ci build_example pr_previews examples_on_main cleanup_pr_previews release"

sync_readme_example:
  uv run python sync_readme_example.py
