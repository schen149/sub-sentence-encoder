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

To load the model and tokenizer:
```python
from model.pl_subencoder import LitSubEncoder
from transformers import AutoTokenizer

model_path = "<your-model-dir>/subencoder_st5_base.ckpt"
model = LitSubEncoder.load_from_checkpoint(model_path)
tokenizer = AutoTokenizer.from_pretrained(model.params.model_name)
```

## Usage
Here are some example ways to use the model.

Suppose you have the following sentences, for which there are a few sub-sentence parts you want to encode. Let's say you express the sub-sentence part in the natural language form. 
```python
sentence1 = "Dracula is a novel by Bram Stoker, published in 1897."
parts1 = ["Dracula is by Bram Stoker", "Dracula is published in 1897"]
sentence2 = "Dracula – a 19th-century Gothic novel, featuring Count Dracula as the protagonist."
parts2 = ["Dracula a 19th-century novel"]
```

Step #1: Align the sub-sentence parts with the original sentence  
```python
from utils.label_utils import align_sub_sentence

spans1 = [align_sub_sentence(full_sent=sentence1, decomp_sent=p) for p in parts1]
spans2 = [align_sub_sentence(full_sent=sentence2, decomp_sent=p) for p in parts2]
```
If needed, you can move the 

Step #2: Tokenize the inputs (and convert the sub-sentence parts into attention mask format)
```python
from utils.data_utils import convert_example_to_features

sent1_ex = [{
  "text": sentence1,
  "spans": spans1
}]

sent2_ex = [{
  "text": sentence2,
  "spans": spans2
}]

sent1_inputs = convert_example_to_features(batch_examples=sent1_ex, tokenizer=tokenizer, max_seq_len=128, sent1_key="text", sent1_props_key="spans")
sent2_inputs = convert_example_to_features(batch_examples=sent2_ex, tokenizer=tokenizer, max_seq_len=128, sent1_key="text", sent1_props_key="spans")
```

The [`convert_example_to_features`](https://github.com/schen149/sub-sentence-encoder/blob/490d3d0a7b625e392dcf031d428267a3a7ca5539/utils/data_utils.py#L158) function returns a [`PropPairModelInput`](https://github.com/schen149/sub-sentence-encoder/blob/490d3d0a7b625e392dcf031d428267a3a7ca5539/utils/data_utils.py#L38) that will be used as input to the model. If you need to move the input on to a different device, e.g. `cuda`, you can do it via

```python
sent1_inputs.to("cuda")
sent2_inputs.to("cuda")
```

Step #3: Encode the examples
```python
sent1_embeddings = model(
    encoder_inputs=sent1_inputs.encoder_inputs,
    num_prop_per_sentence=sent1_inputs.num_prop_per_sentence,
    all_prop_mask=sent1_inputs.all_prop_mask
)

sent2_embeddings = model(
    encoder_inputs=sent2_inputs.encoder_inputs,
    num_prop_per_sentence=sent2_inputs.num_prop_per_sentence,
    all_prop_mask=sent2_inputs.all_prop_mask
)
```
Step #4: See what the cosine similarities between the sub-sentence parts are?
```python
import torch.nn.functional as F

# Between "Dracula is by Bram Stoker" and "Dracula is published in 1897" in sentence 1
sim = F.cosine_similarity(sent1_embeddings[0], sent1_embeddings[1], dim = -1)
print(sim)
# Output: tensor(0.2909)

# Between "Dracula is published in 1897" from sentence 1 and "Dracula a 19th-century novel" from sentence 2
sim = F.cosine_similarity(sent1_embeddings[1], sent2_embeddings[0], dim = -1)
print(sim)
# Output: tensor(0.7180)
```

## Proposition Segmentation
We release the T5-large model + usage instructions for proposition segmentation on Huggingface.  

[https://huggingface.co/sihaochen/SegmenT5-large](https://huggingface.co/sihaochen/SegmenT5-large)


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
