# Release notes

## AxsDB 0.1.2 (*upcoming release*)

* Extended CI matrix to all major OSes (Linux, macOS, Windows) and Python 3.9
  through 3.14 ({ghpr}`13`).
* Fixed `UnicodeDecodeError` when reading `metadata.json` on Windows
  ({ghpr}`13`).
* Added cross-platform coverage path mapping for multi-OS coverage aggregation
  ({ghpr}`13`).
* Added developer installation documentation ({ghpr}`13`).

## AxsDB 0.1.1 (2026-02-18)

* Added a converter to the `AbsorptionDatabase._error_handling_config` attribute
  ({ghpr}`12`).
* Added an error_handling_config argument to the
  `AbsorptionDatabase.from_directory()` constructor ({ghpr}`12`).

## AxsDB 0.1.0 (2026-02-17)

* First beta release. AxsDB is now ready for public release.
