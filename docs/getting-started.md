# Getting Started

## Installation from source 

Clone the repository and install dependencies:

```bash
git clone git@github.com:swat-lab-optimization/ambiegen.git
cd ambiegen
conda create -n ambiegen python=3.10
conda activate ambiegen
pip install -r requirements.txt
```

## Installation with pip

To install AmbieGen using pip, run the following command:

```bash
pip install ambiegen
```

## Usage

### Generating Tests

Run the following to command to generate tests based on the default configuration:

```bash
python generate_tests.py --module-name "ambiegen.testers.uav_tester" --class-name "UAVTester" --runs 3 --config-path "tester_config.yaml"
```

### Comparing Outputs

Use the `compare.py` script to compare results:

```bash
python compare.py  --stats_path "path-to-alg1-stats" "path-to-alg2-stats" --stats_names "alg1" "alg2" --plot_name "my_experiment"
```