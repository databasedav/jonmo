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
      ref = 'Commit "aa827f4804db279f7d22f916522b9f8e7e75d084"
    },
  }
} | std.package.Manifest
