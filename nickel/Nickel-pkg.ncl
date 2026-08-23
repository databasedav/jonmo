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
        ref = 'Commit "7ba85610965b8fc64a7c0c1173f72819daf0da66"
      },
  }
} | std.package.Manifest
