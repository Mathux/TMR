## Note on datasets

Currently, three datasets are widely used for 3D text-to-motion: [KIT-ML](https://motion-annotation.humanoids.kit.edu/dataset/), [HumanML3D](https://github.com/EricGuo5513/HumanML3D) and [BABEL](https://babel.is.tue.mpg.de).

### Unifying the datasets

As explained on their website, [AMASS](https://amass.is.tue.mpg.de) dataset is a large database of human motion unifying different optical marker-based motion capture datasets by representing them within a common framework and parameterization.

Except from a part of HumanML3D which is based on [HumanAct12](https://ericguo5513.github.io/action-to-motion/) (which is also based on [PhSPD](https://drive.google.com/drive/folders/1ZGkpiI99J-4ygD9i3ytJdmyk_hkejKCd?usp=sharing)), almost all the motion data of KIT-ML, HumanML3D and BABEL are included in AMASS.

Currently, the text-to-motion datasets are not compatible in terms of motion representation:
- KIT-ML uses [Master Motor Map](https://mmm.humanoids.kit.edu) (robot-like joints)
- HumanML3D takes motion from AMASS, extract joints using the SMPL layer, rotate the joints (make Y the gravity axis), crop the motions, make all the skeleton similar to a reference, and compute motion features.
- BABEL use raw SMPL parameters from AMASS

To be able to use any text-to-motion dataset with the same representation, I propose in [this repo](https://github.com/Mathux/AMASS-Annotation-Unifier) to unify the datasets, to have the same annotation format. With the agreement of the authors, I included the annotations files in TMR repo, in this folder: [datasets/annotations](datasets/annotations) (for BABEL please follow the instructions). For each datasets, I provide a .json file with:
- The ID of the motion (as found in the original dataset)
- The path of the motion in AMASS (or HumanAct12)
- The duration in seconds
- A list of annotations which contains:
  - An ID
  - The corresponding text
  - The start and end in seconds

Like this one:

```json
{
  "000000": {
    "path": "KIT/3/kick_high_left02_poses",
    "duration": 5.82,
    "annotations": [
      {
        "seg_id": "000000_0",
        "text": "a man kicks something or someone with his left leg.",
        "start": 0.0,
        "end": 5.82
      },
      ...
```

We are now free to use any motion representation.


### Motion representation

Guo et al. uses a representation of motion which includes rotation invariant forward kinematics features, 3D rotations, velocities, foot contacts. Currently, a lot of works in 3D motion generation uses these features. However, these features are not the same for HumanML3D and KIT-ML (not the same number of joints, the scale is different, the reference skeleton is different etc).

To let people use TMR as an evaluator, and be comparable with Guo et al. feature extractor, I propose to process the whole AMASS (+HumanAct12) dataset into the HumanML3D Guo features (which I refer to ``guoh3dfeats`` in the code). Then, we can crop each feature file according to any dataset. I also included the mirrored version of each motions.

### Differences with the released version of HumanML3D
For motion shorter than 10s, this process corresponds to exactly the features file of HumanML3D (example "000000.npy").
As a sanity check, you can verify in python that both .npy corresponds to the same data:

```python
import numpy as np
new = np.load("datasets/motions/guoh3dfeats/humanact12/humanact12/P11G01R02F1812T1847A0402.npy")
old = np.load("/path/to/HumanML3D/HumanML3D/new_joint_vecs/000001.npy")
assert np.abs(new - old).mean() < 1e-10
```

For motion longer than 10s and which are cropped (like "000004.npy"), the results of cropping the features is a bit different than computing the features of the cropped motion. That is because the ``uniform skeleton`` function takes the first frame as reference to compute bone length. However, the difference is quite small.

### Installation
Go to the section "Installation - Set up the datasets" of the [README.md](README.md) to compute the features.


## Credits
For all the datasets, be sure to read and follow their license agreements, and cite them accordingly.

### KIT-ML
```bibtex
@article{Plappert2016,
    author = {Matthias Plappert and Christian Mandery and Tamim Asfour},
    title = {The {KIT} Motion-Language Dataset},
    journal = {Big Data}
    year = 2016
}
```

### HumanML3D
```bibtex
@inproceedings{Guo_2022_CVPR,
    author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
    title     = {Generating Diverse and Natural 3D Human Motions From Text},
    booktitle = {Computer Vision and Pattern Recognition (CVPR)},
    year      = 2022
}
```

### BABEL
```bibtex
@inproceedings{BABEL:CVPR:2021,
  title = {{BABEL}: Bodies, Action and Behavior with English Labels},
  author = {Punnakkal, Abhinanda R. and Chandrasekaran, Arjun and Athanasiou, Nikos and Quiros-Ramirez, Alejandra and Black, Michael J.},
  booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  year = 2021
}
```
