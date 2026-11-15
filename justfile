set allow-duplicate-recipes

KAAJ_HAALKA_COMMIT := "3b496e5d55d86fc57703a90c7c731fe430ec6ac0"

fetch_kaaj_justfile:
  curl https://raw.githubusercontent.com/databasedav/haalka/{{ KAAJ_HAALKA_COMMIT }}/kaaj/justfile > kaaj.just

import? 'kaaj.just'

# nickel format does not yet parse package imports such as `import kaaj`
format_nickels:
  nickel format nickel/Nickel-pkg.ncl

doc *extras:
  cargo +stable doc --all-features --no-deps --locked {{ extras }}

clippy *extras:
  cargo +stable clippy --all-features --all-targets --locked --no-deps {{ extras }} -- --deny warnings

test:
  cargo test tests -- --test-threads=1 && just doctest

# TODO: use an actual list https://github.com/casey/just/issues/2458
exclude_examples := '"test", "utils"'

# TODO: use an actual list https://github.com/casey/just/issues/2458
export_nickels := "ci build_example pr_previews reviewed_pr_preview deploy_reviewed_pr_preview examples_on_main cleanup_pr_previews release"

sync_readme_example:
  uv run python sync_readme_example.py
