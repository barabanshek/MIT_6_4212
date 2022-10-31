# MIT 6.4212, Nikita's repo

This repo contains some stuff I used to paly with while taking 6.4212 at MIT. The stuff also includes the final project in directory `project`. Everythig here is based on the standard Ubuntu 22.04 .

# Setting up this repo

### Prerequisites

0. `Python3` with `venv` and `pip`
1. `Jupyter Notebook`, nice instructions are [here](https://jupyter.org/install)
2. `Drake`, nice instructions are [here](https://drake.mit.edu/pip.html#stable-releases)
3. some extra packages: ``, ``, ``, ``, ``, ``, ``, ``, ``, ``.

### Install the repo and run it

0. `git clone` it
1. `git submodule update --init --recursive` to fetch submolude `manipulation`, i.e. the class [repo](https://github.com/RussTedrake/manipulation) (needed for examples and models).
2. If working in venv, activate them with `source env/bin/activate` or whenever the env is on your machine
2. For local runs, use `jupyter-notebook --no-browser --port=8080`
3. For remote runs, we need to forward some jupyter-notebook and Drake SSH ports, I have a script for this: `run_drake_ssh.sh`. Note that no -X11 required for this to work -- Jupyter works just fine from the local browser when the right ports are forwareded. You might want to change `iptables` sometimes to allow the ports.

### Run the project

To run the project, simply ope the notebook, load the file `project/Main.ipynb` and enjoy it! If any of the cells throughs a package error, simply install it in your python venv.

The README.md inside the `project` folder contains more info and other things on the project itself.
