# AxsDB v1 — Design Document

> Document generated and edited by Vincent Leroy with Claude Sonnet 4.6.
> This is work in progress: there might be mistakes or missing parts.
> I still need to research REPTRAN to understand it completely.

## Glossary

* VMR: Volume mixing ratio, expressed in mol/mol, or its multiples like ppmv.

## Overview

AxsDB v1 provides data handling and evaluation infrastructure for atmospheric
absorption cross-sections, supporting multiple species and data sources. The core
architectural change from v0 is the replacement of a single merged database with a
**component-based design**, where each component encapsulates a single
species/source combination. Components are aggregated by a `Database` object that
orchestrates the full absorption coefficient computation from an atmospheric profile.

```
Database
├── LinearComponent(s)       # one per linear species/source
├── NonlinearComponent(s)    # one per nonlinear species/source
└── SpectralBackend          # LBL | CKD | Reptran | …
```

## Component protocol

Every component, regardless of linearity or spectral mode, implements a uniform
protocol:

```python
class AbstractComponent(ABC):
    @property
    def species(self) -> str: ...

    @property
    def coords(self) -> frozenset[str]:
        """
        Coordinate names this component depends on.
        Always includes 'p' and 'T'; nonlinear components add VMR dims.
        """
        ...

    def lookup(self, atmo: xr.Dataset, **interp_kwargs) -> xr.DataArray:
        """
        Returns absorption cross-section [m²/molecule] on the same
        spatial grid as `atmo`, broadcast-safe.
        """
        ...
```

The `atmo` dataset carries all thermophysical fields on whatever spatial grid the
caller uses (1D pressure levels, 3D lat/lon/alt, etc.). Components are responsible
only for interpolation in `(p, T[, VMR…])` space and are oblivious to spatial
topology. The output dataset is expected to have the same coordinates as the
`atmo` parameter.

## Linear vs nonlinear components

**Linear components** hold a cross-section DataArray with only `(p, T)` dimensions
(plus spectral coordinates). They return `σ(p, T)` directly; the `Database`
applies the VMR scaling. The `vmr_scaled` flag is `False` for linear components.

**Nonlinear components** carry one or more VMR dimensions in their data array
(*e.g.* H₂O self-broadening). They interpolate in VMR-space internally. The
`vmr_scaled` flag is `True`, meaning the `Database` multiplies by number density
only, not again by VMR.

This distinction is encoded in a boolean `vmr_scaled` flag on the component
protocol.

> **Open question: VMR reference for linear components**
>
> The component metadata carries a nullable `vmr_ref` field. The scaling convention
> at evaluation time (*e.g.* whether to apply a first-order correction
> `σ(p,T) × (vmr / vmr_ref)`) is **not yet decided** and must be clarified before
> implementation. **Discuss this with Claudia.**

## Component identity and versioning

Components are identified by a `(species, source, version, spectral_mode)` tuple.
This avoids silent shadowing when multiple components for the same species are
registered (*e.g.* HITRAN2020 vs GEISA for H₂O).

## Spectral backends

The spectral backend is a **coordinate contract** that components satisfy, not a
property of the `Database` as a whole.

| Backend | Spectral coord        | Notes                           |
| ------- |-----------------------| ------------------------------- |
| LBL     | `wavenumber` (float)  | monochromatic, dense            |
| CKD     | `(band, g)`           | band index + g-point            |
| Reptran | `(band, rep)`         | representative wavelength index |
| Custom  | anything              | user-defined                    |

Each component declares its spectral mode via a `spectral_mode` tag. The
`Database` validates consistency at construction time.

### Mixed-backend databases

Mixed-backend support is required, in particular to combine CKD line absorption
with monochromatic continuum data (*e.g.* MT-CKD). The `Database` holds a component
index keyed on `(species, source, version, spectral_mode)` and selects the appropriate
component per species at evaluation time. A priority/fallback chain per species is
supported.

Combining a non-LBL line-absorption component with a monochromatic LBL continuum
requires a dedicated **`SpectralMixer`** abstraction. Only pairings of one
non-LBL mode with LBL are supported; combinations of two non-LBL modes
(*e.g.* REPTRAN + CKD) are physically meaningless and rejected at construction.

An important near-term use case is **REPTRAN line absorption + MT-CKD continuum**.
Because both REPTRAN representative wavelengths and the MT-CKD continuum are
evaluated at explicit wavenumber values, the mixer for this combination reduces
to a pointwise lookup of the continuum at the REPTRAN wavenumber coordinates — no
spectral integration or resampling required. This is the concrete design target
for the first `SpectralMixer` implementation.

## REPTRAN spectral representation

REPTRAN selects a fixed set of **representative wavelengths** per band, each
carrying a quadrature weight. Absorption at each representative point is
monochromatic — no interpolation in the spectral dimension is needed. This
makes REPTRAN the simplest backend to implement and a **first-class production
target** for v1.

```
Dimensions: (band, rep_idx, p, T[, vmr_X, …])
Coordinates:
  band       int or str
  rep_idx    int
  wavenumber float64   (non-dimension coordinate on rep_idx)
  weight     float64   (non-dimension coordinate on rep_idx)
```

A `ReptranComponent` is structurally similar to `CKDDiscreteComponent` (band
index + fixed representative points with weights) but without any g-ordering
constraint. Lookup dispatches to `interp_2d` — the same kernel used by LBL
components — applied independently at each representative point.

> **Open question: REPTRAN resolution variants**
>
> libRadtran distributes REPTRAN data at coarse, medium, and fine resolutions.
> The resolution must be part of the component identity. Encoding it in
> `source` (*e.g.* `"REPTRAN-medium"`) is the simplest option and leaves the
> `(species, source, version, spectral_mode)` tuple unchanged. A dedicated
> `resolution` field is more explicit but adds complexity. **Decide before
> writing the data format spec.**

## CKD g-coordinate: continuous vs discrete

Two representations are supported, implemented as distinct component subtypes
sharing the same protocol.

**Continuous g** (`CKDContinuousComponent`) stores cross-section as a function of
a real-valued `g ∈ [0, 1]` coordinate per band. This is the canonical, maximally
flexible form. The caller chooses the g-point discretisation at evaluation time.

```
Dimensions: (band, g, p, T)
Coordinates:
  band  int or str
  g     float64, values in [0, 1], densely sampled
```

**Discrete g** (`CKDDiscreteComponent`) stores cross-section pre-evaluated at a
fixed set of g-points with associated quadrature weights. The `g` coordinate is
a nominal index; no runtime interpolation in `g` is performed. This is the
production path for Eradiate's hardcoded 16-point scheme.

```
Dimensions: (band, g_idx, p, T)
Coordinates:
  band    int or str
  g_idx   int, 0..15
  g       float64   (non-dimension coordinate on g_idx)
  weight  float64   (quadrature weights, non-dimension coordinate on g_idx)
```

A `CKDDiscreteComponent` can be produced from a `CKDContinuousComponent` via a
factory method, making the continuous data the canonical source of truth:

```python
CKDDiscreteComponent.from_continuous(
    source: CKDContinuousComponent,
    quadrature: Quadrature,   # carries g-points and weights
)
```

> **Note: Alternative to discrete g-points**
>
> Discrete g-points have the disadvantage of requiring to generate a special-purpose
> dataset tailored to the targeted quadrature. Even though this operation is
> not complicated, it requires an additional preprocessing step. As a middle-ground,
> knowing that during computation, the spectral quadrature does not change, we
> can precompute the interpolation weights. For best efficiency, we assume
> constant-step sampling on the g-point dimension.

## Database layer

```python
class Database:
    def __init__(
        self,
        components: Sequence[AbstractComponent],
    ): ...

    def sigma(self, atmo: xr.Dataset, **interp_kwargs) -> xr.DataArray:
        """
        Per-species cross-sections. Returns a DataArray with a 'species'
        dimension alongside spectral and spatial dims.
        """
        ...

    def k_abs(self, atmo: xr.Dataset, **interp_kwargs) -> xr.DataArray:
        """
        Total volumetric absorption coefficient [m⁻¹].
        Shape: (spectral, *spatial).
        """
        ...
```

The `k_abs` computation per component:

```
k_i = σ_i(p, T[, VMR…]) × n(p, T) × vmr_i   # linear (vmr_scaled=False)
k_i = σ_i(p, T, VMR…)   × n(p, T)           # nonlinear (vmr_scaled=True)
```

where `n(p, T)` is the total number density (ideal gas law or better EOS), and
`vmr_i` comes from `atmo`.

## Class hierarchy

The hierarchy is organised along two orthogonal axes: **linearity** (which
gufunc kernel arity the component uses) and **spectral mode** (which spectral
coordinate structure the component satisfies). Supporting types are listed
separately.

```
AbstractComponent (ABC)
├── AbstractLinearComponent (ABC)             vmr_scaled=False; interp_2d kernel
│   ├── LBLComponent                          spectral: wavenumber (float, dense)
│   ├── ReptranComponent                      spectral: (band, rep_idx)
│   ├── CKDContinuousComponent                spectral: (band, g ∈ [0,1])
│   └── CKDDiscreteComponent                  spectral: (band, g_idx)
└── AbstractNonlinearComponent (ABC)          vmr_scaled=True; interp_3d/4d kernel
    ├── LBLNonlinearComponent                 spectral: wavenumber
    ├── ReptranNonlinearComponent             spectral: (band, rep_idx)
    ├── CKDContinuousNonlinearComponent       spectral: (band, g ∈ [0,1])
    └── CKDDiscreteNonlinearComponent         spectral: (band, g_idx)

AbstractSpectralMixer (ABC)
├── PointwiseMixer          REPTRAN/LBL-lines + LBL-continuum (exact wavenumber alignment)
└── BandwiseMixer           CKD-lines + LBL-continuum (continuum collapsed to band mean)
```

### Component classes

```python
@attrs.frozen
class ComponentId:
    species:       str
    source:        str   # e.g. "HITRAN2020", "REPTRAN-medium", "MT-CKD"
    version:       str
    spectral_mode: str   # "lbl" | "reptran" | "ckd_continuous" | "ckd_discrete"


class AbstractComponent(ABC):
    @property
    def id(self) -> ComponentId: ...

    @property
    def species(self) -> str: ...        # shortcut to id.species

    @property
    def vmr_scaled(self) -> bool: ...

    @property
    def coords(self) -> frozenset[str]: ...

    @property
    def spectral_grid(self) -> xr.Dataset:
        """
        Spectral discretization of this component.
        Same format as ``Database.spectral_grid`` for the corresponding mode.
        Used by ``Database`` at construction time to verify grid consistency
        across all primary-mode components.
        """
        ...

    def lookup(
        self,
        atmo: xr.Dataset,
        bounds_policy: BoundsPolicy | None = None,
    ) -> xr.DataArray:
        """Cross-section [m²/molecule] on the same spatial grid as `atmo`."""
        ...


class AbstractLinearComponent(AbstractComponent, ABC):
    """vmr_scaled=False. Data shape: (spectral…, p, T). Uses interp_2d."""
    vmr_scaled: ClassVar[bool] = False


class LBLComponent(AbstractLinearComponent):
    """Spectral dim: wavenumber (float, densely sampled)."""
    ...


class ReptranComponent(AbstractLinearComponent):
    """
    Spectral dims: band, rep_idx.
    Non-dim coords on rep_idx: wavenumber (float), weight (float).
    """
    ...


class CKDContinuousComponent(AbstractLinearComponent):
    """Spectral dims: band, g (float in [0, 1], densely sampled)."""
    ...


class CKDDiscreteComponent(AbstractLinearComponent):
    """
    Spectral dims: band, g_idx (int).
    Non-dim coords on g_idx: g (float), weight (float).
    """
    @classmethod
    def from_continuous(
        cls,
        source: CKDContinuousComponent,
        quadrature: Quadrature,
    ) -> CKDDiscreteComponent: ...


class AbstractNonlinearComponent(AbstractComponent, ABC):
    """vmr_scaled=True. Data shape: (spectral…, p, T, vmr_X[, vmr_Y…])."""
    vmr_scaled: ClassVar[bool] = True

    @property
    def vmr_dims(self) -> tuple[str, ...]:
        """Names of the VMR dimensions in the data array."""
        ...


class LBLNonlinearComponent(AbstractNonlinearComponent):
    """Spectral dim: wavenumber. Adds one or more vmr_X dims."""
    ...


class ReptranNonlinearComponent(AbstractNonlinearComponent):
    """
    Spectral dims: band, rep_idx (same as ReptranComponent).
    Adds one or more vmr_X dims; uses interp_3d / interp_4d kernel.
    """
    ...


class CKDContinuousNonlinearComponent(AbstractNonlinearComponent):
    """Spectral dims: band, g. Adds one or more vmr_X dims."""
    ...


class CKDDiscreteNonlinearComponent(AbstractNonlinearComponent):
    """
    Spectral dims: band, g_idx. Adds one or more vmr_X dims.
    Non-dim coords on g_idx: g (float), weight (float).
    """
    ...
```

### SpectralMixer

```python
class AbstractSpectralMixer(ABC):
    """
    Combines lookup results from components in different spectral modes.
    Invoked by Database when more than one spectral mode contributes to
    a single species (e.g. REPTRAN lines + MT-CKD continuum).
    """
    def mix(
        self,
        contributions: Sequence[xr.DataArray],
    ) -> xr.DataArray: ...


class PointwiseMixer(AbstractSpectralMixer):
    """
    Sums contributions that all carry an explicit `wavenumber` coordinate.
    The continuum is interpolated onto the line-absorption wavenumber grid
    before addition; no spectral resampling or integration is needed.
    Covers REPTRAN-lines + MT-CKD-continuum and LBL + LBL-continuum cases.
    """
    ...


class BandwiseMixer(AbstractSpectralMixer):
    """
    Mixes a band-based component (CKD) with a monochromatic continuum (LBL).
    The continuum has no natural g-point representation, so it is collapsed
    to a single value per band before addition. The collapsing strategy is
    configurable (e.g. band-mean, band-median, or evaluation at band centre).

    Covers CKD-lines + MT-CKD-continuum.
    """
    strategy: str = "band_mean"   # or "band_centre"
    ...
```

### Supporting types

```python
@attrs.frozen
class Quadrature:
    """Quadrature rule for CKD integration."""
    g:       np.ndarray   # shape (n_points,), values in [0, 1]
    weights: np.ndarray   # shape (n_points,), sum to 1
```

`BoundsPolicy` is reused from v0 without changes.

### Database

```python
class Database:
    def __init__(self, components: Sequence[AbstractComponent], cache_size: int = 128):
        """
        Validates components and infers spectral mode and mixer automatically.
        Two checks are performed in order:

        **1. Mode combination check** — the allowed mode sets are:

        * Single mode (any) → used as-is; no mixer needed.
        * ``reptran`` + ``lbl`` → REPTRAN is primary; ``PointwiseMixer`` selected.
        * ``ckd_discrete`` + ``lbl`` → CKD discrete is primary; ``BandwiseMixer`` selected.
        * ``ckd_continuous`` + ``lbl`` → CKD continuous is primary; ``BandwiseMixer`` selected.

        Any other combination raises ``ValueError``.

        **2. Spectral grid consistency check** — all primary-mode components must
        expose the same ``spectral_grid`` (identical dimensions, coordinates, and
        values). Raises ``ValueError`` if grids differ, e.g. two REPTRAN components
        at different resolutions, or CKD components with mismatched band definitions.
        """
        ...

    @property
    def spectral_mode(self) -> str:
        """Inferred spectral mode: 'lbl' | 'reptran' | 'ckd_continuous' | 'ckd_discrete'."""
        ...

    @property
    def spectral_grid(self) -> xr.Dataset:
        """
        Spectral discretization of the database, taken from the shared grid
        of the primary-mode components (guaranteed identical by construction).

        Structure depends on ``spectral_mode``:

        * **LBL** — dimension: ``wavenumber`` [cm⁻¹].
        * **REPTRAN** — dimensions: ``band``, ``rep_idx``;
          non-dimension coordinates: ``wavenumber`` [cm⁻¹], ``weight`` [1].
        * **CKD discrete** — dimensions: ``band``, ``g_idx``;
          non-dimension coordinates: ``g`` [1], ``weight`` [1].
        * **CKD continuous** — dimensions: ``band``, ``g`` [1].
        """
        ...

    def sigma(
        self, atmo: xr.Dataset, bounds_policy: BoundsPolicy | None = None
    ) -> xr.DataArray:
        """
        Per-species cross-sections.
        Shape: (species, spectral…, *spatial).
        """
        ...

    def k_abs(
        self, atmo: xr.Dataset, bounds_policy: BoundsPolicy | None = None
    ) -> xr.DataArray:
        """
        Total volumetric absorption coefficient [m⁻¹].
        Shape: (spectral…, *spatial).
        """
        ...
```

`Database` validates at construction time that all registered components share
the same spectral mode (or that a `mixer` is provided when they do not). It
normalises the `atmo` spatial topology at its boundary and never calls gufuncs
directly — all kernel dispatch is delegated to `lookup`.

## Interpolation and performance

### Requirements

`xarray.interp()` is insufficient for production performance. The interpolation
layer must be implemented using custom gufuncs, consistent with the approach
already taken in AxsDB v0.

### Gufunc kernel family

The gufunc layer is built around a small set of **fixed-arity interpolation kernels**,
each corresponding to a coordinate signature:

```
interp_2d(p, T, p_grid, T_grid, data)              # linear components
interp_3d(p, T, v, p_grid, T_grid, v_grid, data)   # nonlinear, 1 VMR
interp_4d(...)                                     # nonlinear, 2 VMRs
```

Each kernel operates in a **transformed coordinate space** (*e.g.* log-p, linear-T).
The coordinate transform is applied to the grid at data load time, keeping the
gufunc logic simple and the physics conventions in the data preparation layer.

Each component's `lookup` method is responsible for applying the coordinate
transform to query points before dispatching to the appropriate kernel. The
`Database` never calls gufuncs directly.

### CKD continuous g

For continuous-g components, the `g` interpolation dimension would naively break
separability. However, since `g` is queried at the same fixed quadrature points
for every atmospheric layer, interpolation weights in `g` can be precomputed once
per band and reused, reducing the problem to a weighted sum over precomputed slices.

### Mixed spatial topologies

The `Database` normalises the `atmo` dataset at its boundary (flatten spatial
dims → call gufuncs → reshape output). This keeps all gufunc implementations
simple and topology-agnostic.

### Bounds handling

The bounds-handling policy interface from AxsDB v0 (dict-based, per-dimension
configuration) is reused directly and passed as an argument to `lookup`.

### Extensibility

This design allows a Numba implementation to be replaced by a C extension (via
nanobind) without touching the database logic, since the performance-critical
code is fully localized in the component `lookup` methods.

## Data formats

### Cross-section component file

One file per component (NetCDF4 for CKD/Reptran, Zarr for large LBL datasets).

```
Dimensions: (spectral_coord(s), p, T[, vmr_X, …])
Coordinates:
  p        [Pa]   pressure levels
  T        [K]    temperature grid
  vmr_X    [1]    VMR for species X (nonlinear components only)
  <spectral coordinates per backend>
Attributes:
  species:       str
  source:        str           # e.g. "HITRAN2020", "MT-CKD"
  version:       str
  spectral_mode: str           # "lbl" | "ckd" | "reptran"
  vmr_ref:       float | null  # open question: scaling convention TBD
  vmr_scaled:    bool
  convention:    str           # "absorption" (σ in m²/molecule)
  axsdb_version: str
```

For CKD discrete-g files, `g` and `weight` are stored as non-dimension coordinates
on `g_idx`.

For REPTRAN files, `wavelength` and `weight` are stored as non-dimension coordinates
on `rep_idx`. The `wavelength` values are the actual spectral positions of each
representative point (not indices), enabling direct lookup by the `SpectralMixer`.

### Atmospheric profile (`atmo` dataset)

```
Variables:
  p        (*spatial)  [Pa]
  T        (*spatial)  [K]
  n        (*spatial)  [m⁻³]   optional; computed from p, T if absent
  vmr_X    (*spatial)  [1]     for each species X
Attributes:
  spatial_dims: list[str]      # e.g. ["z"] or ["lat", "lon", "z"]
  crs: str                     # e.g. "cartesian_1d" | "spherical_shell"
```

### Absorption coefficient output

```
Dimensions: (spectral_coord(s), *spatial_dims)
Attributes:
  spectral_mode: str
  species:       list[str]
  units:         "m-1"
  database_id:   str           # hash or tag of the Database configuration
```

## Open questions

- **VMR reference convention for linear components**: scaling behaviour at
  evaluation time is not yet decided.
- **`BandwiseMixer` collapsing strategy**: the method for reducing a monochromatic
  continuum to a per-band scalar for CKD (band-mean, band-centre evaluation, or
  g-weighted average) has accuracy implications and should be validated against
  reference calculations before finalising.
- **REPTRAN resolution variants**: encode resolution in `source`
  (*e.g.* `"REPTRAN-medium"`) or as a dedicated identity field. Decide before
  writing the data format spec.
