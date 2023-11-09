# Sub-Sentence-Encoder
The official code repo for "Sub-Sentence Encoder: Contrastive Learning of Propositional Semantic Representations".

[https://arxiv.org/pdf/2311.04335.pdf](https://arxiv.org/pdf/2311.04335.pdf)

Contrary to sentence embeddings, where the meaning of each sentence is represented by one fixed-length vector, sub-sentence-encoder produces distinct embeddings for different parts of a sentence.  

<img src="https://github.com/schen149/sub-sentence-encoder/blob/main/figure/teaser.png" alt="" data-canonical-src="https://github.com/schen149/sub-sentence-encoder/blob/main/figure/teaser.png" width="400" />

## Installation
The model is implemented with `pytorch_lightning`. See `requirements.txt` for the list of required packages.

We will release a native pytorch + huggingface compatible version of the model soon. 

## Model Checkpoint
The model checkpoints of different variants of the models can be found in this [google drive folder](https://drive.google.com/drive/folders/179Ga1WElV3yjxIA5MRk9lxRo5b9ImDqN?usp=sharing).

To load the model:
```python
from model.pl_subencoder import LitSubEncoder

model_path = "<your-model-dir>/subencoder_st5_base.ckpt"
model = LitSubEncoder.load_from_checkpoint(model_path)
```

## Citation
```
@article{chen2023subsentence,
  title={Sub-Sentence Encoder: Contrastive Learning of Propositional Semantic Representations},
  author={Sihao Chen and Hongming Zhang and Tong Chen and Ben Zhou and Wenhao Yu and Dian Yu and Baolin Peng and Hongwei Wang and Dan Roth and Dong Yu},
  journal={arXiv preprint arXiv:2311.04335},
  year={2023},
  URL = {https://arxiv.org/pdf/2311.04335.pdf}
}
```
