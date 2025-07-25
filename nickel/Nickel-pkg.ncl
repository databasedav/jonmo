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
      ref = 'Commit "27104c4f47980cb53ec64c6d5e7874cf4f3caaec"
    },
  }
} | std.package.Manifest
