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
      ref = 'Commit "7064230a1c6f7f1b562301ed3d51cf2f8f8a9a01"
    },
  }
} | std.package.Manifest
