# TODO list

- Documentation
  - Docstrings for all functions and methods of public API
  - Short informal docstrings for tests (i.e. How is the functionality tested?)
  - Include citations to original papers
  - Format: [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html)?
  - Citation format wie [hier](https://github.com/biotite-dev/hydride/blob/4b5a1c4348cf8f9b0878c1480a09ffcf101cba48/src/hydride/relax.pyx#L649-L659)?
- Example scripts with short applications for `examples` directory
  - Contact switch-off -> Jan
  - Covariance matrix from MD simulation -> speak with Kay
- B factors
  - GNM und ANM -> Faisal
  - Tests (e.g. comparison with ProDy) -> Faisal
- Force fields
  - Docstrings for predefined force fields (static methods) including citations
    in `TypeSpecificForceField` -> Jan
  - Test predefined force fields with multi-chain structure and reference interaction matrix from
    ProDy, Bio3d or BioPhysConnector -> Faisal
  - Generalization to more types of molecules
    (e.g. Nucleic acids, small molecules)? -> Patrick
  - More force fields with different concept, for example based on
    secondary structure