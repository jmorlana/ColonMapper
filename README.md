# ColonMapper: topological mapping and localization for colonoscopies

ColonMapper is a topological mapping and localization algorithm able to build a topological map of the whole colon autonomously. In a second stage, ColonMapper can localize another sequence of the same patient against the previously built topological map. 

<p align="center">
  <a><img src="assets/colonmapper_logo.png" width="95%"/></a>
</p>

## Installation

To install the ColonMapper environment, simply use conda as:

``` bash
conda create -n colonmapper --file requirements.txt
```

Paths are defined in [`settings.py`](settings.py). There you should define where you store the data and the evaluation logs.
Folder structure should resemble this:

```
.
├── datasets     # endomapper and C3VD
├── models       # trained models to run experiments
└── logs         # where results are stored
    └── model 1  # example model
      ├── experiment 1   
      └── experiment 2  
```

## Download trained models and evaluation data

Before starting, you have to download the trained models, which can be found [here](https://unizares-my.sharepoint.com/:f:/g/personal/684222_unizar_es/EgaHCLbx4oBPtpuP1bdu460BwH9ZSF-GuQhf_tD1XNWVfQ?e=ufxjyJ).

The images used for evaluation, both for Endomapper and C3VD, can be found [here](https://unizares-my.sharepoint.com/:f:/g/personal/684222_unizar_es/Egvbdh5CngxCreGZXz0vjpkBs0XuzDVCaitfuz87_WYr0w?e=ubQe5M).


## Usage

To run the topological mapping as described in the paper, run the following command:

```bash
cd ColonMapper
python mapping_vg.py --resume=[PATH_TO_MODELS]/resnet50conv4_netvlad_0_0_640_hard_resize_layer2/best_model.pth --datasets 027 035 cross C3VD
```

To run the Bayesian localization against canonical maps as described in the paper, run the following command:

```bash
cd ColonMapper
python localization_vg.py --experiment_name=bayesian_reject --resume=[PATH_TO_MODELS]/resnet50conv4_netvlad_0_0_640_hard_resize_layer2/best_model.pth --datasets 027 entry_cross cross C3VD --bayesian --threshold_probability=0.5 --reject_outliers --reject_strategy=diffusion
```


## Related Publication:

Javier Morlana, Juan D. Tardós and J.M.M. Montiel, **ColonMapper: topological mapping and localization for colonoscopies**, *ICRA 2024*. [PDF](https://arxiv.org/pdf/2305.05546)
```
@inproceedings{morlana2024colonmapper,
  title={ColonMapper: topological mapping and localization for colonoscopy},
  author={Morlana, Javier and Tard{\'o}s, Juan D and Montiel, JMM},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={6329--6336},
  year={2024},
  organization={IEEE}
}
```





