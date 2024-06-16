# StyleDrop + Post-Filtering


This is a modified version (unofficial) of [StyleDrop: Text-to-Image Generation in Any Style](https://arxiv.org/abs/2306.00983).

Round 2 and many utilities are added.

Codes only for research purpose

The Detailed Instruction for usage, refer to [link](https://github.com/aim-uofa/StyleDrop-PyTorch), the original unnofficial Pytorch implementation of StyleDrop




## Round 2 Training

use [custom_IT.py](./configs/cuustom_IT.py) for iterative training

#### Data Selection
```bash
       streamlit run image_selector.py
```


## Inference

use demo codes in [notebook](Basic_tools.ipynb) to calculate Text/Style score or to generate image using adapter of your taste!

## Citation
```bibtex
@article{sohn2023styledrop,
  title={StyleDrop: Text-to-Image Generation in Any Style},
  author={Sohn, Kihyuk and Ruiz, Nataniel and Lee, Kimin and Chin, Daniel Castro and Blok, Irina and Chang, Huiwen and Barber, Jarred and Jiang, Lu and Entis, Glenn and Li, Yuanzhen and others},
  journal={arXiv preprint arXiv:2306.00983},
  year={2023}
}
```


## Acknowlegment

* The implementation is based on [MUSE-PyTorch](https://github.com/baaivision/MUSE-Pytorch)
* Many thanks for the generous help from [Zanlin Ni](https://github.com/nzl-thu)
* Repository originally shared publically from [StyleDrop-PyTorch](https://github.com/aim-uofa/StyleDrop-PyTorch)

