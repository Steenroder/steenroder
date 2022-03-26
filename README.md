# steenroder


The widespread use in applied topology of the barcode of filtered cellular complexes rests on a balance between discriminatory power and computability. It has long been envision that the strength of this invariant could be increase using cohomology operations. This package computes the recently defined *Sq*<sup>*k*</sup>-barcodes which have been shown to effectively increase the discriminatory power of barcodes on real-world data.


For a complete presentation of these invariants please consult [Persistence Steenrod modules](https://arxiv.org/abs/1812.05031) by U. Lupo, A. Medina-Mardones and G. Tauzin.


## License

``steenroder`` is distributed under the [MIT license](https://github.com/Steenroder/steenroder/LICENSE).


## Documentation

Please visit https://steenroder.github.io/steenroder/ and navigate to the version you are interested in.


## Installation

### Dependencies


The latest stable version of ``steenroder`` requires:

- python (>= 3.8)
- numpy (>= 1.19.1)
- numba (>= ?)
- psutils (>= ?)
- gudhi (>= ?)
- plotly (>= ?)

To run the examples, jupyter is required.

### Installation

The simplest way to install ``steenroder`` is using ``pip``   ::

    python -m pip install -U steenroder

If necessary, this will also automatically install all the above dependencies. Note: we recommend
upgrading ``pip`` to a recent version as the above may fail on very old versions.


## Contributing

We welcome new contributors of all experience levels. The Steenroder
community goals are to be helpful, welcoming, and effective. To learn more about
making a contribution to ``steenroder``, please consult `the relevant page
https://github.com/Steenroder/steenroder/CONTRIBUTING.md.

### Testing

After developer installation, you can launch the test suite from outside the
source directory:

    pytest steenroder


## Important links

- Official source code repo: https://github.com/Steenroder/steenroder
- Download releases: https://pypi.org/project/steenroder/
- Issue tracker: https://github.com/Steenroder/steenroder/issues


## Citing steenroder

If you use ``steenroder`` in a scientific publication, we would appreciate citations to the following paper:

   [Persistence Steenrod modules](https://arxiv.org/abs/1812.05031)

You can use the following BibTeX entry:

    @article{steenroder,
           author = {{Lupo}, Umberto and {Medina-Mardones}, Anibal M. and {Tauzin}, Guillaume},
            title = "{Persistence Steenrod modules}",
          journal = {arXiv e-prints},
    archivePrefix = {arXiv},
           eprint = {1812.05031},
     primaryClass = {math.AT},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2018arXiv181205031L},
    }

## Contacts


steenroder@gmail.com
