# Skeleton Codebase for RL Experiments (Hydra • Submitit • MariaDB)

A lightweight, extensible template for running large-scale reinforcement learning (RL) experiments with structured configuration, sweep execution (local or Slurm), and automatic logging to MySQL/MariaDB. Notebooks are provided for result retrieval and visualization. 

---

**Jump to:** [At a glance](#at-a-glance) · [Install](#installation) · [Config](#configuration) · [Run](#running-experiments) · [Analysis](#results--analysis) · [Repo Structure](#repository-structure) · [Envs](#adding-environments) · [Contributing](#contributing) · [License](#license) · [Citation](#citation)

---

## At a glance

| Topic             | Where                              | Notes                                                                                                                                      |
| ----------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Configuration** | `configs/`                         | Hydra configs for env/agent/sweep/DB. Set `db_prefix` on CC.                                                                               |
| **Algorithms**    | `agents/`                          | greedy\_ac, mpo, ppo, sac, td3, tsallis\_awac. See [Supported Algorithms](#supported-algorithms).                                          |
| **Policies**      | `models/policy_parameterizations/` | beta, gaussian, q\_gaussian, softmax, squashed\_gaussian, student. See [Policy Parameterizations](#policy-parameterizations).              |
| **Logging**       | MySQL/MariaDB                      | Schema: `configs/schema/default-schema.yaml`; credentials: `configs/db/credentials.yaml`.                                                  |
| **Run (local)**   | `main.py`                          | Joblib sweep: `python main.py -m -cn config_joblib_local run=0`; single run: remove `-m`. See [Running Experiments](#running-experiments). |
| **Run (Slurm)**   | submitit configs                   | Example: `python main.py -m -cn config_submitit_cedar run=0`; remove `-m` for a single run.                                                |
| **Analysis**      | `notebooks/`                       | Example notebooks to retrieve & visualize results.                                                                                         |


## Requirements

* **Python 3.10 or newer.** Some features rely on Python ≥3.10.
* A running **MySQL/MariaDB** server you can connect to (local or remote).

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

#### Database credentials

Add your DB credentials to [`configs/db/credentials.yaml`](configs/db/credentials.yaml).

You can log results to a local database, to a Google Cloud VM, or to a Compute Canada (CC) database server by following these [instructions](https://docs.computecanada.ca/wiki/Database_servers).

**Tip:** Create a `credentials-local.yaml` and add it to `.gitignore` so you don’t commit secrets.

#### Database prefix (Compute Canada)

Set `db_prefix` to your username in [`configs/config.yaml`](configs/config.yaml). This is required on CC, where database names must begin with your CC username.

#### Database schema

The schema used for logging is defined in [`configs/schema/default-schema.yaml`](configs/schema/default-schema.yaml).

## Running Experiments

#### Quick test (single run)

Run a single experiment by omitting the sweep flag:

```bash
python main.py -cn config_joblib_local run=0
```

#### Local sweep (joblib)

Run a sweep of experiments as defined in [`configs/config_joblib_local.yaml`](configs/config_joblib_local.yaml). The `args.run` argument is the run ID for the first experiment; subsequent experiments are assigned `args.run + sweep_id` in the database.

```bash
python main.py -m -cn config_joblib_local run=0
```

If you’re running on a machine **without Slurm**, use the joblib configuration files.

#### Slurm sweep (submitit)

Use the submitit config for your cluster (e.g., `config_submitit_cedar`). Example sweep:

```bash
python main.py -m -cn config_submitit_cedar run=0
```

To schedule a **single** run on Slurm, remove the `-m` flag.

#### Run IDs and database primary key

The `run` field defined in the config files is the **primary key** in the database. Within a single database, this key **must be unique** or the experiment will not start.

You can ensure uniqueness by either:

* Passing a unique `run` value for each experiment, or
* Using a **separate database** for each sweep.

## Results & Analysis

All run statistics are logged to your configured database. See the example notebooks under [`notebooks/`](notebooks/) for how to retrieve and visualize results.

## Repository Structure

```
agents/                             # Agent implementations and interfaces
configs/                            # Hydra configs (env, db, sweeps, etc.)
environments/                       # Environment helper
experiment/                         # Experiment manager and logger
models/                             # Policy / value networks
models/policy_parameterizations/    # Policy parameterization modules
notebooks/                          # Example analysis notebooks
utils/                              # Common utilities
main.py                             # Entry point for sweeps and single runs
main_ppo.py                         # Entry point for PPO
save_policy_evolution.py            # Entry point for saving policy snapshots
```

## Supported Algorithms

Algorithms implemented under `agents/`:

* **Greedy Actor–Critic** (`greedy_ac`) — [Neumann et al., 2018](https://arxiv.org/abs/1810.09103)
* **MPO** — Maximum a Posteriori Policy Optimization (`mpo`) — [Abdolmaleki et al., 2018](https://arxiv.org/abs/1806.06920)
* **PPO** — Proximal Policy Optimization (`ppo`) — [Schulman et al., 2017](https://arxiv.org/abs/1707.06347) — see [`main_ppo.py`](main_ppo.py) for an entry point
* **SAC** — Soft Actor–Critic (`sac`) — [Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)
* **TD3** — Twin Delayed DDPG (`td3`) — [Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477)
* **Tsallis AWAC** (`tsallis_awac`) — [Zhu et al., 2024](https://openreview.net/pdf?id=HNqEKZDDRc)

## Policy Parameterizations

Policy heads/modules under `models/policy_parameterizations/`:

* **beta** — Beta distributions (bounded policy)
* **gaussian** — Clipped Gaussian policies
* **q\_gaussian** — q-Gaussian policies
* **softmax** — Categorical (discrete) policies
* **squashed\_gaussian** — Tanh-squashed Gaussian
* **student** — Student‑t policies with learnable dof (heavy‑tailed)

## Adding Environments

If your environment is a **Gymnasium** environment, you only need to add a config file under [`configs/env/`](configs/env). The environment will be initialized via the factory in [`environments/factory.py`](environments/factory.py).

## Troubleshooting

If your code is not running or logging properly, the chances are that it is due to a problem with the database. Many of these errors may be silent. Try one of the following:

* **DB connection errors** → Double‑check `configs/db/credentials.yaml` (or your local override), network access/ports, and the `db_prefix` on CC.
* **Duplicate run IDs** → Use a fresh `run` or a separate database per sweep.
* **Schema errors** → Ensure that whatever you are logging is defined properly in the schema file. The datatypes and the order of entries matter.

## Contributing

Contributions are welcome!

* **Bugs & features:** Please open an issue to report bugs or request features. For larger changes, start a discussion first to align on scope and design.
* **Pull requests:** Keep PRs focused and well-described. Where applicable, update docs/examples and include a minimal reproduction if you change experiment logic.

## Citation

This codebase was used for online RL experiments in our ICLR 2025 paper **[q-Exponential Family For Policy Optimization](https://proceedings.iclr.cc/paper_files/paper/2025/file/6507b115562bb0a305f1958ccc87355a-Paper-Conference.pdf).**
If you found our work helpful in your research and need a reference, then you may use the following bibtex:

```bibtex
@inproceedings{ICLR2025_6507b115,
 author = {Zhu, Lingwei and Shah, Haseeb and Wang, Han and Nagai, Yukie and White, Martha},
 booktitle = {International Conference on Representation Learning},
 editor = {Y. Yue and A. Garg and N. Peng and F. Sha and R. Yu},
 pages = {40717--40744},
 title = {q-exponential family for policy optimization},
 url = {https://proceedings.iclr.cc/paper_files/paper/2025/file/6507b115562bb0a305f1958ccc87355a-Paper-Conference.pdf},
 volume = {2025},
 year = {2025}
}
```
