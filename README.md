# DeepONet repository for WRR submission

### Author: Alex Sun

This repository includes codes for WRR paper, 

Bridging hydrological ensemble simulation and learning using deep neural operators.


- `config_uns_uq_multi.yaml`: configuration file for generating Figures 3-5 in the manuscript
- `uppersink_deeponet_uq_multisite.py`: python code for generating Figures 3-5 in the manuscript
- `config_uns_uq_multi_rev.yaml`: configuration file for generating Figures S1 and S2 in the supporting information
- `uppersink_deeponet_uq_multisite_rev.py`: python code for generating Figures S1 and S2 in the supporting information
- `uq_deeponet_main.py`: main code for training deeponet seq2seq model
- `uq_deeponet.py`: DeepONet model for seq2seq

Data parser is `readensemble.py`.

---

### Requirements

Install Modulus 22.09.

Put the codes in deeponet and start the docker container as

```console
#!/bin/bash
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
           --runtime nvidia -p 5001:6006 -v ${PWD}/deeponet:/examples \
           -it --rm nvcr.io/nvidia/modulus/modulus:22.09  bash
```

After entering the container, install the dependencies

```
pip install hydrostats
pip install deap
pip install seaborn
pip install mpl-scatter-density
```

### Run code

```
python uppersink_deeponet_uq_multisite_rev.py
```

The learning tasks are controlled inside `uppersink_deeponet_uq_multisite_rev.py`. 

- itask=1, train deeponet surrogate
- itask=2, run genetic algorithm (GA)
- itask=3, do UQ
- itask=4, plot GA results for the five experiments
- itask=5, compare the effect of ensemble size

The GA experiments are controlled by expno in `uppersink_deeponet_uq_multisite_rev.py`.

The train ensemble size experiment are controlled by train_split_no in `config_uns_uq_multi_rev.yaml`.


