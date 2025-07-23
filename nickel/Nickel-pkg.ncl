let KAAJ_HAALKA_COMMIT = "854d87b07177e4dd986c16f9e04028b24fe156b6" in
{
  name = "jonmo_ci",
  version = "0.1.0",
  authors = ["databasedav <31483365+databasedav@users.noreply.github.com>"],
  minimal_nickel_version = "1.12.0",
  description = "github actions ci generation for jonmo",
  dependencies = {
    kaaj = 'Git { url = "https://github.com/databasedav/haalka", path = "kaaj", ref = 'Commit KAAJ_HAALKA_COMMIT },
  }
} | std.package.Manifest
