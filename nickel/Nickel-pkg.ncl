{
  name = "jonmo_ci",
  version = "0.1.0",
  authors = ["databasedav <31483365+databasedav@users.noreply.github.com>"],
  minimal_nickel_version = "1.12.2",
  description = "github actions ci generation for jonmo",
  dependencies = {
    kaaj = 'Git {
      url = "https://github.com/databasedav/haalka",
      path = "kaaj",
      ref = 'Commit "460bc00514cc61b5476c8a5028ea22b1e6a02d10"
    },
  }
} | std.package.Manifest
