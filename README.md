# MULTIPHADE (MULti-TIp PHAse DEtection)

This is the repository of the MULti-TIp PHAse DEtection (MULTIPHADE) Software.

This repository is structured as follows:

* doc: any documentations of MULTIPHADE (developing, testing, application, etc.) is collected here
* dataio: H5 file handling tools (reader/writer)
* python_tests: temporary python scripts, e.g. for performance comparison
* schemadef: JSON schemas
* tests: unit tests, feature tests, model tests
* tools: python scripts for building and testing
* Pipfile: Definition of the python environment via *pipenv*
* sby.py: Stochastic Bubble Generator (SBG)
* sbg_functions.py: functions used by the SBG
* sbg_plot.py: functions used by the SBG for data vizualization
* mssrc.py: Multi-Sensor Signal ReConstructor (MSSRC)
* velocity_tsa.py: tool for velocity time series analysis
