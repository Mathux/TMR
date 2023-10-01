<div align="center">

# TMR: Text-to-Motion Retrieval
## Using Contrastive 3D Human Motion Synthesis

<a href="https://mathis.petrovich.fr"><strong>Mathis Petrovich</strong></a>
·
<a href="https://ps.is.mpg.de/~black"><strong>Michael J. Black</strong></a>
·
<a href="https://imagine.enpc.fr/~varolg"><strong>G&#252;l Varol</strong></a>


[![ICCV2023](https://img.shields.io/badge/ICCV-2023-9065CA.svg?logo=ICCV)](https://iccv2023.thecvf.com)
[![arXiv](https://img.shields.io/badge/arXiv-TMR-A10717.svg?logo=arXiv)](https://arxiv.org/abs/2305.00976)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

</div>


## Description
Official PyTorch implementation of the paper:
<div align="center">

[**TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis**](https://arxiv.org/abs/2305.00976).

</div>

Please visit our [**webpage**](https://mathis.petrovich.fr/tmr/) for more details.

### Bibtex
If you find this code useful in your research, please cite:

```bibtex
@inproceedings{petrovich23tmr,
    title     = {{TMR}: Text-to-Motion Retrieval Using Contrastive {3D} Human Motion Synthesis},
    author    = {Petrovich, Mathis and Black, Michael J. and Varol, G{\"u}l},
    booktitle = {International Conference on Computer Vision ({ICCV})},
    year      = 2023
}
```
and if you use the re-implementation of TEMOS of this repo, please cite:

```bibtex
@inproceedings{petrovich22temos,
    title     = {{TEMOS}: Generating diverse human motions from textual descriptions},
    author    = {Petrovich, Mathis and Black, Michael J. and Varol, G{\"u}l},
    booktitle = {European Conference on Computer Vision ({ECCV})},
    year      = 2022
 }
```

You can also put a star :star:, if the code is useful to you.

## Installation :construction_worker:

<details><summary>Create environment</summary>
&emsp;

Create a python virtual environnement:
```bash
python -m venv ~/.venv/TMR
source ~/.venv/TMR/bin/activate
```

Install [PyTorch](https://pytorch.org/get-started/locally/)
```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then install remaining packages:
```
python -m pip install -r requirements.txt
```

which corresponds to the packages: pytorch_lightning, einops, hydra-core, hydra-colorlog, orjson, tqdm, scipy.
The code was tested on Python 3.10.12 and PyTorch 2.0.1.

</details>

<details><summary>Set up the datasets</summary>

### Introduction
The process is a little bit different than other repos because we need to have a common reprensenation for HumanML3D, KITML and BABEL (to be able to train on one, and evaluate on another).
If you are currious about the details, I recommand you to read this file: [DATASETS.md](DATASETS.md). I also put the bibtex files of the datasets, which I recommand you to cite.

### Get the data
Please follow the instructions of the ``raw_pose_processing.ipynb`` of the [HumanML3D](https://github.com/EricGuo5513/HumanML3D) repo, to get the ``pose_data`` folder.
Then copy or symlink the pose_data folder in ``datasets/motions/``:
```bash
ln -s /path/to/HumanML3D/pose_data datasets/motions/pose_data
```

### Compute the features
Run the following command, to compute the HumanML3D Guo features on the whole AMASS (+HumanAct12) dataset.

```bash
python -m prepare.compute_guoh3dfeats
```

It should process the features (+ mirrored version) and saved them in ``datasets/motions/guoh3dfeats``.


### Compute the text embeddings
Run this command to compute the sentence embeddings and token embeddings used in TMR for each datasets.

```bash
python -m prepare.text_embeddings data=humanml3d
```

This will save:
- the token embeddings of ``distilbert`` in ``datasets/annotations/humanml3d/token_embeddings``
- the sentence embeddings of ``all-mpnet-base-v2`` in ``datasets/annotations/humanml3d/sent_embeddings``


### Compute statistics (already done for you)

To get statistics of the motion distribution for each datasets, you can run the following commands. It is already included in the repo, so you don't have to. The statistics are computed on the training set.

```bash
python -m prepare.motion_stats data=humanml3d
```

It will save the statistics (``mean.pt`` and ``std.pt``) in this folder ``stats/humanml3d/guoh3dfeats``. You can replace ``data=humanml3d`` with ``data=kitml`` or ``data=babel`` anywhere in this repo.

</details>

## Training :rocket:

```bash
python train.py [OPTIONS]
```

<details><summary>Details</summary>
&emsp;

By default, it will train TMR on HumanML3D and store the folder in ``outputs/tmr_humanml3d_guoh3dfeats`` which I will call ``RUN_DIR``.
The other options are:

#### Models:
- ``model=tmr``: TMR (by default)
- ``model=temos``: TEMOS

#### Datasets:
- ``data=humanml3d``: HumanML3D (by default)
- ``data=kitml``: KIT-ML
- ``data=babel``: BABEL

</details>

<details><summary>Extracting weights</summary>
&emsp;

After training, run the following command, to extract the weights from the checkpoint:

```bash
python extract.py run_dir=RUN_DIR
```

It will take the last checkpoint by default. This should create the folder ``RUN_DIR/last_weights`` and populate it with the files: ``motion_decoder.pt``, ``motion_encoder.pt`` and ``text_encoder.pt``.
This process makes loading models faster, it does not depends on the file structure anymore, and each module can be loaded independently. This is already done for pretrained models.

</details>

## Pretrained models :dvd:

```bash
bash prepare/download_pretrain_models.sh
```

This will put pretrained models in the ``models`` folder.
Currently, there is:
- TMR trained on HumanML3D with Guo et al. humanml3d features ``models/tmr_humanml3d_guoh3dfeats``

More models will be available later on.

## Evaluation :bar_chart:

```bash
python retrieval.py run_dir=RUN_DIR
```

It will compute the metrics, show them and save them in this folder ``RUN_DIR/contrastive_metrics/``.


## Usage :computer:

### Encode a motion
Note that the .npy file should corresponds to HumanML3D Guo features.

```bash
python encode_motion.py run_dir=RUN_DIR npy=/path/to/motion.npy
```

### Encode a text

```bash
python encode_text.py run_dir=RUN_DIR text="A person is walking forward."
```

### Compute similarity between text and motion
```bash
python text_motion_sim.py run_dir=RUN_DIR text=TEXT npy=/path/to/motion.npy
```
For example with ``text="a man sets to do a backflips then fails back flip and falls to the ground"`` and ``npy=HumanML3D/HumanML3D/new_joint_vecs/001034.npy`` you should get around 0.96.


## Launch the demo

### Encode the whole motion dataset
```bash
python encode_dataset.py run_dir=RUN_DIR
```


### Text-to-motion retrieval demo
Run this command:

```bash
python app.py
```

and then open your web browser at the address: ``http://localhost:7860``.

## Localization (WIP)

The code will be available a bit later.


### Reimplementation of TEMOS (WIP)

<details><summary>Details and difference</summary>
&emsp;

[TEMOS code](https://github.com/Mathux/TEMOS) was probably a bit too abstract and some users struggle to understand it. As TMR and TEMOS share a similar architecture, I took the opportunity to rewrite TEMOS in this repo [src/model/temos.py](src/model/temos.py) to make it more user friendly. Note that in this repo, the motion representation is different from the original TEMOS paper (see [DATASETS.md](DATASETS.md) for more details). Another difference is that I precompute the token embeddings (from distilbert) beforehand (as I am not finetunning the distilbert for the final model). This makes the training around x2 faster and it is more memory efficient.

The code and the generations are not fully tested yet, I will update the README with pretrained models and more information later.

</details>


## License :books:
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including PyTorch, PyTorch3D, Hugging Face, Hydra, and uses datasets which each have their own respective licenses that must also be followed.