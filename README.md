# EvoLlama

This is the official repository for the paper *EvoLlama: Enhancing LLMs' Understanding of Proteins via Multimodal Structure and Sequence Representations*.

\[Dataset\] | \[[Model](https://huggingface.co/nwliu/EvoLlama)\] | \[[Preprint](https://arxiv.org/abs/2412.11618)\]

# Quickstart

## Environment Setups

We recommend using Python >= 3.9, and then simply use pip to install the required packages:

```shell
pip install -r requirements.txt
```

## Download Model Weights

Model weights are publicly available on [🤗HuggingFace](https://huggingface.co/nwliu/EvoLlama). 
During training, the parameters of Llama-3 are frozen. To initialize EvoLlama, 
you need to manually download the LLM weights at [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

For projection-tuned EvoLlama, only the projection layers are trainable. 
Therefore, you need to manually download the [ProteinMPNN weights](https://github.com/dauparas/ProteinMPNN/blob/main/vanilla_model_weights/v_48_020.pt)/
[GearNet weights](https://zenodo.org/records/7593637/files/mc_gearnet_edge.pth?download=1),
and the [ESM-2 weights](https://huggingface.co/facebook/esm2_t33_650M_UR50D).

The table below provides a summary of the EvoLlama model family and includes links to their model weights on 🤗HuggingFace.

|             Models             |         Stages         |  Datasets   |     PDB     |                                                   Links                                                   |
|:------------------------------:|:----------------------:|:-----------:|:-----------:|:---------------------------------------------------------------------------------------------------------:|
| EvoLlama (ProteinMPNN + ESM-2) |   Projection Tuning    |  SwissProt  | AlphaFold-2 |   [Download](https://huggingface.co/nwliu/EvoLlama/tree/main/projection_tuning/protein_mpnn_esm2_650m)    |
| EvoLlama (ProteinMPNN + ESM-2) | Supervised Fine-tuning | PMol + PEER |   ESMFold   | [Download](https://huggingface.co/nwliu/EvoLlama/tree/main/supervised_fine_tuning/protein_mpnn_esm2_650m) |
|   EvoLlama (GearNet + ESM-2)   |   Projection Tuning    |  SwissProt  | AlphaFold-2 |                                                 Download                                                  |
|   EvoLlama (GearNet + ESM-2)   | Supervised Fine-tuning | PMol + PEER |   ESMFold   |                                                 Download                                                  |

## Inference

Helper functions for initializing EvoLlama and generating responses are provided in `src/infer/infer.py`. Note that
the function `infer()` accepts a list of lists of PDB files and sequences, and a list of arbitrary prompts as inputs.
When utilizing EvoLlama without a structure/sequence encoder, set the corresponding parameter `None`.

```python
import os
from src.infer.infer import init_evo_llama, infer

# 1. Initialize EvoLlama
model_weights_path = '/path/to/EvoLlama'
llm_weights_path = '/path/to/llm'

evo_llama = init_evo_llama(
    structure_encoder_path=os.path.join(model_weights_path, 'structure_encoder_weights'),
    structure_encoder_name='ProteinMPNN',
    sequence_encoder_path=os.path.join(model_weights_path, 'sequence_encoder'),
    llm_path=llm_weights_path,
    projection_path=os.path.join(model_weights_path, 'projection_weights.bin'),
    projection_fusion=True,
    is_inference=True
)

# 2. Inference with EvoLlama
pdb_files = ['examples/ea91f233142ab1a17749be765a461255.pdb']  # We use the MD5 hash of the protein sequence as the filename.
sequences = ['MANHKSTQKSIRQDQKRNLINKSRKSNVKTFLKRVTLAINAGDKKVASEALSAAHSKLAKAANKGIYKLNTVSRKVSRLSRKIKQLEDKI']
prompt = 'Analyze the given amino acid sequence, and determine the function of the resulting protein, its subcellular localization, and any biological processes it may be part of.'

responses = infer(evo_llama, [pdb_files], [sequences], [prompt])
```

Additionally, simply run scripts `scripts/eval_molinst.sh` and `scripts/eval_peer.sh` to evaluate EvoLlama
on the protein understanding and protein property prediction tasks, respectively.

```shell
# Evaluate EvoLlama on the protein understanding tasks.
bash scripts/eval_molinst.sh

# Evaluate EvoLlama on the protein property prediction tasks.
bash scripts/eval_peer.sh
```

## Training

Coming soon ...

# Citation

```bibtex
@misc{liu2024evollama,
    title={EvoLlama: Enhancing LLMs' Understanding of Proteins via Multimodal Structure and Sequence Representations}, 
    author={Nuowei Liu and Changzhi Sun and Tao Ji and Junfeng Tian and Jianxin Tang and Yuanbin Wu and Man Lan},
    year={2024},
    eprint={2412.11618},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2412.11618}, 
}
```
