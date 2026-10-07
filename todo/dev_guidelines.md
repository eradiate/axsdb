# Development guidelines for v1 refactoring

- Push Pint to interface boundaries. All compute happens in fixed units.
- Push xarray to interface boundaries. All data is served as plain Numpy arrays or views. xarray is still allowed to hold data, but is never invoked in compute code.
