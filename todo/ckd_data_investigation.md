# Correlated-k / Absorption Database Structures — Comparative Notes

> **AI Disclosure**: This is the result of a research query with Claude Opus 4.8.
> Generated, reviewed and edited by Vincent Leroy.

Scope: how molecular absorption data is tracked, parameterised, interpolated,
combined and stored across CKDMIP, ecCKD, REPTRAN (libRadtran), Eradiate/AxsDB,
and RTMOM.

## Sources consulted

- CKDMIP (ECMWF Confluence; Hogan & Matricardi 2020, GMD 13, 6501–6521).
- ecCKD (ECMWF "How CKD tools work").
- REPTRAN (Gasteiger et al. 2014, JQSRT 148, 99–115; libRadtran docs).
- Eradiate / AxsDB: repo `eradiate/axsdb` (LGPL-3.0); AxsDB docs (Ac-v1 format);
  Eradiate docs; Eradiate GMD paper (gmd.copernicus.org/articles/19/4289/2026).
- RTMOM: Govaerts, RTMOM V0B.13 User's Manual, Rayference, RAY 01, April 2017.

## Layered distinction (CKDMIP)

1. Reference line-by-line datasets (LBLRTM outputs) — the data "analysed".
2. CKD models (e.g. ecCKD) generated from (1).
REPTRAN and RTMOM are single-layer analogues of (2); AxsDB sits closest to (2).

## Quantity tracked

- CKDMIP ref: spectral optical depth per gas (cross-section per molecule as invariant).
- ecCKD: molar absorption coefficient (cross-section per mole).
- REPTRAN: absorption cross-section Cabs.
- AxsDB released (Ac-v1): volume absorption coefficient `sigma_a` [m^-1] (mixture-level).
- RTMOM: per-gas k-coefficient, labelled "Abs. Coeff."; dimensionally [cm^2/mol]
  (multiplies number density q [mol/cm^3] to give optical thickness) → cross-section-like.

## Concentration variable

- All models: mole fraction / volume mixing ratio (not mass fraction).
- RTMOM: profiles in ppmv → converted to number density q = c·rho_air·1e-6 [mol/cm^3];
  total-column rescaling for H2O [kg/m^2], O3 [DU], CO2 [ppmv surface].

## Tabulation axes / state dependence

- CKDMIP Idealized: p, T, H2O mole fraction (53 p × 6 T).
- ecCKD: (p,T) for well-mixed; (p,T,xH2O) for H2O (nonlinear axis).
- REPTRAN: (p,ΔT) for all; explicit xH2O axis for H2O.
- AxsDB Ac-v1: w, p, t, x_<species> (per-species); CKD adds w(bin), wbv, g, ng.
- RTMOM: AGT schema carries (pressure, concentration, temperature) per gas, BUT
  shipped data use n_conc = 1 → effectively (P, T) only. Three radiative entities:
  H2O, O3, and one "well-mixed" composite (CO2, CH4, N2O, CO, O2). No nonlinear
  H2O concentration axis; water enters via q_H2O and a separate continuum term.

## Interpolation and combination

- ecCKD: LUT interp in (p,T[,xH2O]); sum optical depths per g-point; coefficients
  globally re-optimised against Evaluation-1 fluxes/heating rates.
- REPTRAN: linear interp in (p,T,xH2O); weighted mean over representative wavelengths.
- AxsDB: interp in transformed coords (Numba gufunc interp_2d/3d/4d); CKD quadrature
  over g (16-pt Gauss default); species handled via joint mixture tabulation.
- RTMOM: linear interp in (P,T). Transmittance = triple product of Gauss-quadrature
  sums over H2O, O3, well-mixed (Eq. 4.41). RTE solved per (η,ζ,ξ) triplet, indexed
  by k with N_k = N_H2O·N_O3·N_wmg (Eq. 4.42); per-term τ additive (Eq. 4.44).
  Layer total τ_L = τ_gas + τ_scat + τ_Rayleigh (Eq. 4.62). Optional EFFECTIVE mode
  (Eq. 4.46): column-conservative, not layer-conservative.

## Data format and layout

- CKDMIP ref: NetCDF4/HDF5, compressed, one gas per file (+ concentration sidecars); ~1 TB.
- ecCKD: single self-describing NetCDF definition file.
- REPTRAN: NetCDF, one file per species (+ representative-λ/weight files).
- AxsDB Ac-v1: a database is a DIRECTORY — N spectral-chunk NetCDF files +
  index.csv + spectral.csv + metadata.json; xarray in memory. Distributed as tar.gz
  (codenames: gecko, komodo [mono]; monotropa, mycena, panellus, tuber [CKD]).
- RTMOM: MFD (MOM Formatted Database) directory of per-category flat files
  (SSR, QUA, SDD, AGT, AVP, SPA, IRR). Most ASCII; AGT is BINARY, RECL = MAX_TERM ×
  sizeof(float) = 200. AGT size (records) = 14 + n_gas·(4 + n_pres·(1 + n_conc·(1 + n_temp))).
  Files named AGT_<method>_<sname>_<nnnn>.MFD (default method REGL1). Spectral
  discretisations: LOWRES 10 nm, MEDRES 5 nm, HIGRES 1 nm (manual prints "µm" — a
  units typo; values are nm, consistent with the 2022 paper).

## Summary table

| Aspect           | CKDMIP ref (LBL)   | ecCKD                | REPTRAN                | AxsDB Ac-v1                       | RTMOM V0B.13                                 |
| ---------------- | ------------------ | -------------------- | ---------------------- | --------------------------------- | -------------------------------------------- |
| Quantity         | Optical depth/gas  | Molar abs. coeff.    | Cross-section Cabs     | Volume abs. coeff. sigma_a [m^-1] | k-coeff [cm^2/mol]                           |
| Conc. var.       | Mole fraction      | Mole fraction ψ      | Mole fraction / xH2O   | Mole fraction x_species           | ppmv → q [mol/cm^3]                          |
| Axes             | p, T, xH2O         | (p,T);(p,T,xH2O) H2O | (p,ΔT);+xH2O H2O       | w,p,t,x_species (+g CKD)          | schema (P,conc,T); shipped (P,T)             |
| Conc. dependence | Explicit           | Linear;nonlinear H2O | Linear;explicit xH2O   | Tabulated per species             | Linear for all 3 entities                    |
| Interpolation    | n/a                | LUT                  | Linear (p,T,xH2O)      | Transformed-coord                 | Linear (P,T)                                 |
| Combination      | Sum τ              | Sum τ per g-point    | Sum τ; wtd mean rep. λ | Joint mixture; g-quad             | Product 3 entities; sum τ + aer + Rayleigh   |
| Format           | NetCDF4/HDF5       | Single NetCDF        | NetCDF                 | Dir + NetCDF + 3 sidecars         | MFD dir; AGT binary                          |
| File count       | Many (gas×dataset) | One                  | One/species            | Many + sidecars                   | One AGT/(method×SDD×interval) + per-category |

## Key contrasts (RTMOM vs AxsDB)

- RTMOM stores a per-molecule (cross-section-like) coefficient; AxsDB Ac-v1 stores a
  mixture volume coefficient.
- RTMOM has a concentration axis in schema but ships it collapsed (n_conc = 1); no
  nonlinear H2O tabulation, unlike CKDMIP/REPTRAN.
- RTMOM uses a legacy fixed-record binary format (AGT); AxsDB uses NetCDF/xarray.
