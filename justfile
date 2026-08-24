set allow-duplicate-recipes

KAAJ_HAALKA_COMMIT := "76f9d4427977a475624a90c397e682a37931be51"

fetch_kaaj_justfile:
  curl https://raw.githubusercontent.com/databasedav/haalka/{{ KAAJ_HAALKA_COMMIT }}/kaaj/justfile > kaaj.just

import? 'kaaj.just'

# nickel format does not yet parse package imports such as `import kaaj`
format_nickels:
  nickel format nickel/Nickel-pkg.ncl

test:
  cargo test tests -- --test-threads=1 && just doctest

# TODO: use an actual list https://github.com/casey/just/issues/2458
exclude_examples := '"test", "utils"'

# TODO: use an actual list https://github.com/casey/just/issues/2458
export_nickels := "ci build_example pr_previews reviewed_pr_preview deploy_reviewed_pr_preview examples_on_main cleanup_pr_previews release"

sync_readme_example:
  uv run python sync_readme_example.py
