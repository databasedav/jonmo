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
        ref = 'Commit "3b496e5d55d86fc57703a90c7c731fe430ec6ac0"
      },
  }
} | std.package.Manifest
