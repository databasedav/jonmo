{
  name = "jonmo_ci",
  version = "0.1.0",
  authors = ["databasedav <31483365+databasedav@users.noreply.github.com>"],
  minimal_nickel_version = "1.12.2",
  description = "github actions ci generation for jonmo",
  dependencies = {
    kaaj =
      'Git {
        url = "https://github.com/databasedav/haalka",
        path = "kaaj",
        ref = 'Commit "76f9d4427977a475624a90c397e682a37931be51"
      },
  }
} | std.package.Manifest
