<div align="center">

# TMR: Text-to-Motion Retrieval
## Using Contrastive 3D Human Motion Synthesis

</div>

## Description
Official PyTorch implementation of the paper [**"TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis"**](https://arxiv.org/abs/2305.00976).

Please visit our [**webpage**](https://mathis.petrovich.fr/tmr/) for more details.


### Bibtex
If you find this code useful in your research, please cite:

```
@article{petrovich23tmr,
    title     = {{TMR}: Text-to-Motion Retrieval Using Contrastive {3D} Human Motion Synthesis},
    author    = {Petrovich, Mathis and Black, Michael J. and Varol, G{\"u}l},
    journal   = {arXiv preprint},
    year      = {2023}
}
```

You can also put a star :star:, if the code is useful to you.


### Inference code
(Training code will arrive soon)


## Installation :construction_worker:
### 1. Create the environnement
This code is tested on Python 3.10.

```bash
python -m venv tmr_env
source tmr_env/bin/activate
pip install -r requirements.txt
```

### 2. Launch the demo

This command will download the necessary weights and launch the demo in a URL accessible by a web browser.

```bash
python app.py
```
