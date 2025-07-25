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
      ref = 'Commit "be880c3f68e8e5378f8d5dc7c7769d86e4f2ca5d"
    },
  }
} | std.package.Manifest
