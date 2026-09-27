# Release notes

## AxsDB 0.2.0 (*upcoming release*)

* Exposed `units` module as public API ({ghpr}`17`).
* Reduced the per-call overhead of `eval_sigma_a_mono()` and
  `eval_sigma_a_ckd()`: the coordinate grids of each data file are now stored
  with the cached interpolation plan instead of being read through xarray on
  every call.
* `CKDAbsorptionDatabase.eval_sigma_a_ckd()` no longer copies the selected
  spectral bin on every call: only the bin and the two g-points bracketing the
  query are read. In lazy mode, it no longer loads the entire `sigma_a`
  variable into memory on the first call.
* Sped up `MonoAbsorptionDatabase.eval_sigma_a_mono()` about 4-fold by
  replacing `xarray.DataArray.interp()` on the spectral dimension with
  interpolation between the two bracketing spectral slices.
* Fixed the rebuild of a missing `spectral.csv` file, which always failed with
  a `TypeError`.

## AxsDB 0.1.2 (2026-02-18)

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
