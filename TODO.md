# AxsDB TODO list

- Error handling
  - [ ] Review exceptions and improve them
- Testing
  - [ ] Add integration tests for per-bound error handling
  - [x] Add doctest execution
- Interpolation
  - [ ] Consider parallel execution for independent dimension groups
  - [ ] Investigate numba parallel option for gufuncs
  - [ ] Cache compiled numba functions more aggressively
  - [ ] Consider supporting additional interpolation methods (nearest, cubic)
