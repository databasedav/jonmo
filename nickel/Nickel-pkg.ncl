let KAAJ_HAALKA_COMMIT = "5e06f448c3515fa0dc1cb9ad66fcc0192756fdaf" in
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
