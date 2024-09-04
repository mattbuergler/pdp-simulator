# Phase-Detection Probe Simulator for Turbulent Bubbly Flows

This is the repository of the Phase-Detection Probe Simulator for Turbulent Bubbly Flows (pdp-sim) Software.

This repository is structured as follows:

- **doc:** Any documentation of pdp-sim (development, testing, application, etc.) is collected here.
- **dataio:** H5 file handling tools (reader/writer).
- **schemadef:** JSON schemas.
- **tests:** Unit tests, feature tests, model tests.
- **tools:** Python scripts for building and testing.
- **Pipfile:** Definition of the Python environment via *pipenv*.
- **stsg_ssg.py:** Stochastic Timeseries Generator and Synthetic Signal Generator (STSG-SSG).
- **stsg_ssg_functions.py:** Functions used by the STSG-SSG.
- **mssp.py:** Multi-Sensor Signal Processing (MSSP).
- **run_pdp_sim.py:** Script for running the various steps of a phase-detection probe simulations.

## Getting Started

### Prerequisites

The pdp-sim requires the following dependencies:
- python==3.11.0
- numpy==2.1.1
- pandas==2.2.2
- h5py==3.11.0
- joblib==1.4.2
- jsonschema==4.23.0
- matplotlib==3.9.2
- pathlib==1.0.1
- scipy==1.14.1
- statsmodels==0.14.2

### Installation

To install pdp-sim, follow these steps:

1. Clone this repository to your local machine:

    ```bash
    git clone https://gitlab.ethz.ch/vaw/public/pdp-sim.git
    ```

2. Navigate to the cloned directory:

    ```bash
    cd pdp-sim
    ```

We recommend running pdp-sim in a virtual python environment using pipenv. The installation of pipenv is described in the [documentation](doc/user/setup_python_environment.md).

3. Install the required dependencies using pipenv:

    ```bash
    pipenv install
    ```

4. Activate the virtual environment:

    ```bash
    pipenv shell
    ```

5. You're ready to use pdp-sim!

## Usage

pdp-sim can be used for simulating phase-detection probe measurements in turbulent bubbly flows.

### Running the code

In order to run a simulation of phase-detection probe measurements in turbulent bubbly flows, create a new folder for the simulation and add a JSON configuration file specifying the flow properties, probe characteristics and signal post-processing algorithm. Have a look at the examples under [tests](tests) for the structure of the configuration file.

pdp-sim is based on a modular workflow. The basic steps of a simulation include:

- Generating a 3-D stochastic velocity time series based on the Langevin equations.
- Generating the synthetic signal by tracking the movement of bubbles with respect to the sensors of the phase-detection probe
- Running a signal processing algorithm to recover flow properties, such as velocities and void fractions.

A detailed description of the workflow can be found in the peer-reviewed publication of [Bürgler et al. (2024)](#references).

#### Generating a 3-D stochastic velocity time series

A 3-D stochastic velocity time series can be generated with the Stochastic Time Series Generation and Synthetic Signal Generation (STSG-SSG) Python script `stsg_ssg.py`, using the `timeseries` keyword for the `run` flag (-r). The path to the simulation folder containing the configuration JSON file (config.json) must be provided via command line argument.

```shell
python stsg_ssg.py -r timeseries path/to/simulation
```

#### Generating the synthetic signal

The synthetic signal can be generated with the Stochastic Time Series Generation and Synthetic Signal Generation (STSG-SSG) Python script `stsg_ssg.py`, using the `signal` keyword for the `run` flag (-r). The path to the simulation folder containing the configuration JSON file (config.json) must be provided via command line argument.

```shell
python stsg_ssg.py -r signal path/to/simulation
```

#### Processing the synthetic signal

The synthetic signal can be processed with the Multi-Sensor Signal Processing (MSSP) Python script `mssp.py`. The path to the simulation folder containing the configuration JSON file (config.json) must be provided via command line argument.

```shell
python mssp.py path/to/simulation
```

## Support

For support, bug reports, or feature requests, please open an issue in the [issue tracker](https://gitlab.ethz.ch/vaw/public/pdp-sim/-/issues) or contact Matthias Bürgler at <buergler@vaw.baug.ethz.ch>.


## Authors and acknowledgment

This software is developed by Matthias Bürgler in collaboration and under the supervision of Dr. Daniel Valero, Dr. Benjamin Hohermuth, Dr. David F. Vetsch and Prof. Dr. Robert M. Boes. Matthias Bürgler and Dr. Benjamin Hohermuth were supported by the Swiss National Science Foundation (SNSF) [grant number 197208].
The code is inspired by previously developed stochastic bubble generators ([Bung & Valero, 2017](#references); [Valero et al., 2019](#references); [Kramer, 2019](#references); [Kramer et al., 2019](#references); [Bürgler et al., 2022](#references)).

## Copyright notice

(c)2024 ETH Zurich, Matthias Bürgler, Daniel Valero, Benjamin Hohermuth, David F. Vetsch, Robert M. Boes, D-BAUG, Laboratory of Hydraulics, Hydrology and Glaciology (VAW)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.


## References

Bung, D. B., & Valero, D. 2017. FlowCV-An open-source toolbox for computer vision applications in turbulent flows. In *Proceedings 37th IAHR World Congress*, Kuala Lumpur, Malaysia, pp. 5356-5365. https://hdl.handle.net/2268/214198

Bürgler, M., Hohermuth, B., Vetsch, D. F., & Boes, R. M. 2022. Comparison of Signal Processing Algorithms for Multi-Sensor Intrusive Phase-Detection Probes. In *Proceedings 39th IAHR World Congress*, Granada, Spain, International Association for Hydro-Environment Engineering and Research, pp. 5094-5103. https://doi.org/10.3929/ethz-b-000566288

Bürgler, M., Valero, D., Hohermuth, B., Boes, R.M., & Vetsch, D.F. 2024. Uncertainties in measurements of bubbly flows using phase-detection probes. International Journal of Multiphase Flow 181, 104978. https://doi.org/10.1016/j.ijmultiphaseflow.2024.104978

Kramer, M. 2019. Particle size distributions in turbulent air-water flows. In *E-Proceedings of the 38th IAHR World Congress*, pp. 5722-5731. https://doi.org/10.3850/38WC092019-0680

Kramer, M., Valero, D., Chanson, H., & Bung, D. B. 2019. Towards reliable turbulence estimations with phase-detection probes: an adaptive window cross-correlation technique. *Experiments in Fluids*, 60(1), 2. https://doi.org/10.1007/s00348-018-2650-9

Valero, D., Kramer, M., Bung, D.B., & Chanson, H. 2019. A stochastic bubble generator for air water flow research. In *E-Proceedings of the 38th IAHR World Congress*. Panama City, Panama, pp. 5714–5721. https://doi.org/10.3850/38WC092019-0909

## Citation

If you use this package in academic work, please consider citing our work:

Bürgler M., Valero D., Hohermuth B., Boes R.M., & Vetsch D.F. 2024.
Phase-detection probe simulator for turbulent bubbly flows. ETH Zurich. https://gitlab.ethz.ch/vaw/public/pdp-sim

We recommend to also cite the peer-review publication:

Bürgler, M., Valero, D., Hohermuth, B., Boes, R.M., & Vetsch, D.F. 2024. Uncertainties in measurements of bubbly flows using phase-detection probes. *International Journal of Multiphase Flow*. (in Press). https://doi.org/10.1016/j.ijmultiphaseflow.2024.104978
