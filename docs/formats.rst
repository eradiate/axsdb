Data formats
============

AxsDB reads data from several formats, documented in this page.

Absorption Coefficient Format v1 (Ac v1)
----------------------------------------

These formats store the absorption coefficient contribution by each radiatively
active species as a function of a spectral coordinate. Two variants of this
format are defined, specializing for two different spectral representations of
the radiative transfer simulation problem. Both variants share a common
directory structure and auxiliary file organization, but differ in their NetCDF
data layout:

* **Monochromatic (Ac-Mono)**: Dense wavelength grids for monochromatic
  calculations.
* **Correlated-k distribution (Ac-CKD)**: Band-averaged data with quadrature
  points for efficient band calculations.

These formats were historically designed for the Eradiate radiative transfer
model and have been applied fixes to work around critical performance issues.
As a result, some design decisions may look unintuitive. Addressing the flaws of
the chosen data layout is foreseen in a new iteration of this database format.

Common structure and metadata
-----------------------------

Both formats organize data as a directory containing:

.. code-block:: text

   <dataset_name>/
   ├── index.csv           # File-to-spectral-range mapping
   ├── spectral.csv        # Detailed wavelength information
   ├── metadata.json       # Dataset provenance and parameters
   └── <data_files>.nc     # NetCDF data files

Data files typically split the spectral dimension among multiple NetCDF files,
and each data file is not required to contain information for all species:
radiatively inactive species are omitted.

The potentially large number of NetCDF data files can make spectral lookups
inefficient if no specific optimization is applied. To accelerate this process,
two index files (which can be generated from the data files) have been added to
the original specification:

Index file (``index.csv``)
    Maps data files to their spectral coverage. Columns:

    * ``filename``: NetCDF file name
    * ``wn_min [cm^-1]``, ``wn_max [cm^-1]``: Wavenumber bounds
    * ``wl_min [nm]``, ``wl_max [nm]``: Wavelength bounds

    .. code-block:: text
        :caption: Example

        filename,wn_min [cm^-1],wn_max [cm^-1],wl_min [nm],wl_max [nm]
        dataset-295_305.nc,32786.88,33898.30,295.0,305.0
        dataset-305_315.nc,31746.03,32786.88,305.0,315.0
        ...

Spectral file (``spectral.csv``)
    Lists all spectral points in the dataset and maps them to the data file
    that contains them. Columns:

    * ``filename``: NetCDF file containing this spectral point
    * ``wavelength [nm]``: Central wavelength
    * ``wbound_lower [nm]``, ``wbound_upper [nm]``: Wavelength bounds (empty for
      monochromatic data)

    .. code-block:: text
        :caption: Example (Monochromatic)

        filename,wavelength [nm],wbound_lower [nm],wbound_upper [nm]
        dataset.nc,345.00048,,
        dataset.nc,345.00169,,
        ...

    .. code-block:: text
        :caption: Example (CKD)

        filename,wavelength [nm],wbound_lower [nm],wbound_upper [nm]
        dataset-295_305.nc,299.91,295.0,305.0
        dataset-305_315.nc,309.91,305.0,315.0
        ...

In addition, a metadata file contains arbitrary metadata in the JSON format.

Metadata file (``metadata.json``)
    JSON file documenting dataset provenance, generation parameters, and references.
    Structure:

    .. code-block:: text
        :caption: Example

        {
          "description": "Dataset description",
          "version": {
            "codename": "dataset_name_v1",
            "short_codename": "dataset_v1"
          },
          "history": {
            "date_created": "YYYY-MM-DD",
            "date_started": "YYYY-MM-DD",
            "date_completed": "YYYY-MM-DD",
            "by": "Creator name or source"
          },
          "spectral_range": { ... },
          "pressure_grid": { ... },
          "temperature_grid": { ... },
          "mole_fraction_grid": { ... },
          "references": [ ... ]
        }

Data file format (Ac-Mono v1)
-----------------------------

Format
    ``xarray.Dataset`` (in-memory), NetCDF (storage)

Dimensions
    * ``w``: radiation wavelength (typically thousands of points)
    * ``p``: air pressure
    * ``t``: air temperature
    * ``x_<species>``: mole fraction values for each absorbing species
      (*e.g.* ``x_CO2``, ``x_CH4``, ``x_O2``)

Coordinates
    *All dimension coordinates; \
    when relevant, units are required and specified in the "units" metadata field.*

    * ``w(w)`` float [length]
    * ``p(p)`` float [pressure]
    * ``t(t)`` float [temperature]
    * ``x_<species>(x_species)`` float [dimensionless]

Data variables
    *When relevant, units are required and  specified in the "units" metadata field.*

    * ``sigma_a(w, p, t, x_<species>...)``: volume absorption coefficient [length^-1]

Data file format (Ac-CKD v1)
----------------------------

Format
    ``xarray.Dataset`` (in-memory), NetCDF (storage)

Dimensions
    * ``w``: bin central wavelength (typically 1 per file, representing the band)
    * ``wbv``: bin bound (size 2, lower and upper)
    * ``g``: cumulative probability of the absorption coefficient distribution
    * ``ng``: number of quadrature points for error estimation
    * ``p``: air pressure
    * ``t``: air temperature
    * ``x_<species>``: mole fraction values for each absorbing species
      (*e.g.* ``x_CO2``, ``x_CH4``, ``x_O2``)

Coordinates
    *All dimension coordinates; when relevant, units are required and specified
    in the "units" metadata field.*

    * ``w(w)`` float [length]
    * ``wbv(wbv)`` str
    * ``g(g)`` float [dimensionless]
    * ``ng(ng)`` int [dimensionless]
    * ``p(p)`` float [pressure]
    * ``t(t)`` float [temperature]
    * ``x_<species>(x_<species>)`` float [dimensionless]

Data variables
    *When relevant, units are required and  specified in the "units" metadata field.*

    * ``sigma_a(w, p, t, x_<species>...)``: volume absorption coefficient [length^-1]
    * ``wbounds(w, wbv)``: spectral bin bound values [length]
    * ``error(w, ng)``: relative error on transmittance when using the quadrature,
      optional [dimensionless]
