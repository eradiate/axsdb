:hide-toc:
:layout: landing

AxsDB documentation
===================

**Date**: |today| | **Version**: |version|

AxsDB provides an interface to the absorption cross section databases
of the `Eradiate radiative transfer model <https://eradiate.eu>`_\ .

.. grid:: 1 1 2 3
    :gutter: 2
    :padding: 0

    .. grid-item-card:: :iconify:`material-symbols:book-2 height=1.5em` Docs
        :link: getting_started
        :link-type: doc

        Read about AxsDB and its API.

    .. grid-item-card:: :iconify:`material-symbols:description height=1.5em` API reference
        :link: api/axsdb
        :link-type: doc

        Browse the API reference.

    .. grid-item-card:: :iconify:`material-symbols:code height=1.5em` Developer guide
        :link: dev/installation
        :link-type: doc

        Contribute to and maintain AxsDB.

    .. grid-item-card:: :iconify:`mdi:clock height=1.5em` Changelog
        :link: changelog
        :link-type: doc

        Release history.

    .. grid-item-card:: :iconify:`simple-icons:github height=1.5em` GitHub
        :link: https://github.com/eradiate/axsdb/

        Browse the source code.

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Usage

    getting_started
    formats
    error_handling

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: API reference

    api/axsdb

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Development

    dev/installation
    dev/benchmarking
    dev/interpolation
    Typing <dev/typing>
    dev/release

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: About

    changelog
