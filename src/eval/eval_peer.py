import fire
import os
import sys

import json
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.infer.infer import init_evo_llama, infer
from src.dataset.multimodal import MultimodalDataset, collate_fn


TASK_NAMES = [
    'solubility',
    'subcellular_localization',
    'binary_localization',
    'fold',
    'yeast_ppi',
    'human_ppi'
]

ZERO_SHOT_PROMPT_ENHANCEMENT = {
    'solubility': '\n\n' + r"""Choose "Soluble." or "Not soluble." to answer the question.""",
    'subcellular_localization': '\n\n' + r"""Choose "Cell membrane." or "Cytoplasm." or "Endoplasmic reticulum." or "Golgi apparatus." or "Lysosome/Vacuole." or "Mitochondrion." or "Nucleus." or "Peroxisome." or "Chloroplast." or "Extracellular." to answer the question.""",
    'binary_localization': '\n\n' + r"""Choose "Membrane-bound." or "Soluble." to answer the question.""",
    'fold': r"""Output the answer using an integer from 0 to 1194, e.g., "0.".""",
    'yeast_ppi': '\n\n' + r"""Choose "Yes." or "No." to answer the question.""",
    'human_ppi': '\n\n' + r"""Choose "Yes." or "No." to answer the question.""",
}

DATASET_TYPES = ['training', 'validation', 'test']


def evaluate_peer(
    model,
    task_name: str,
    dataset_dir: str,
    pdb_dir: str,
    dataset_type: str,
    save_dir: str,
    save_result: bool = False,
    batch_size: int = 1,
    use_structure: bool = True,
    use_sequence: bool = True,
    is_zero_shot: bool = False
):
    assert task_name in TASK_NAMES
    assert dataset_type in DATASET_TYPES

    dataset_path = f'{dataset_dir}/peer_{task_name}_{dataset_type}.json'
    eval_dataset = MultimodalDataset(dataset_path, pdb_dir)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    pred = []
    truth = []
    for i, eval_data in enumerate(tqdm(eval_dataloader, desc=task_name)):
        pdb_files = eval_data['pdb_files']
        sequences = eval_data['sequences']
        if not use_structure:
            pdb_files = None
        if not use_sequence:
            sequences = None
        prompts = [prompt + '\n' + ZERO_SHOT_PROMPT_ENHANCEMENT[task_name] for prompt in eval_data['prompts']] if is_zero_shot else eval_data['prompts']
        pred_responses = infer(model, pdb_files, sequences, prompts)

        pred.extend([response.strip() for response in pred_responses])
        truth.extend([res.strip() for res in eval_data['responses']])

    if save_result:
        json_results = []
        idx = 1
        for pred, ref in zip(pred, truth):
            json_results.append({
                'id': idx,
                'prediction': pred,
                'reference': ref
            })
            idx += 1
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(f'{save_dir}/{task_name}.json', 'w') as f:
            json.dump(json_results, f, indent=4)

    return accuracy_score(truth, pred)


def evaluate(
    task_name: str,
    dataset_dir: str,
    pdb_dir: str,
    dataset_type: str,
    save_dir: str,
    save_result: bool = False,
    batch_size: int = 1,
    structure_encoder_path: str = None,
    structure_encoder_name: str = None,
    sequence_encoder_path: str = None,
    llm_path: str = None,
    projection_path: str = None,
    projection_fusion: bool = False,
    is_inference: bool = True,
    use_flash_attention: bool = False,
    use_lora: bool = True,
    lora_adapter_path: str = None,
    sequence_encoder_max_length: int = 2048,
    llm_max_length: int = 4096,
    structure_embedding_dim: int = 128,
    sequence_embedding_dim: int = 1280,
    llm_embedding_dim: int = 4096,
    protein_max_length: int = 2048,
    use_structure: bool = True,
    use_sequence: bool = True,
    is_zero_shot: bool = False
):
    evo_llama = init_evo_llama(
        structure_encoder_path,
        structure_encoder_name,
        sequence_encoder_path,
        llm_path,
        projection_path,
        projection_fusion,
        is_inference,
        use_flash_attention,
        use_lora,
        lora_adapter_path,
        sequence_encoder_max_length,
        llm_max_length,
        structure_embedding_dim,
        sequence_embedding_dim,
        llm_embedding_dim,
        protein_max_length
    )
    evo_llama.eval()

    if task_name == 'all':
        for task_name in TASK_NAMES:
            try:
                performance = evaluate_peer(
                    evo_llama, task_name, dataset_dir, pdb_dir, dataset_type, save_dir, save_result, batch_size, use_structure, use_sequence, is_zero_shot
                )
                print(f'Task {task_name}: {performance}')
            except Exception as e:
                print(e)
    else:
        try:
            performance = evaluate_peer(
                evo_llama, task_name, dataset_dir, pdb_dir, dataset_type, save_dir, save_result, batch_size, use_structure, use_sequence, is_zero_shot
            )
            print(f'Task {task_name}: {performance}')
        except Exception as e:
            print(e)


if __name__ == '__main__':
    fire.Fire(evaluate)
