import json
import fire
import os
import sys
from evaluate import load
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.infer.infer import init_evo_llama, infer
from src.dataset.multimodal import MultimodalDataset, collate_fn


TASK_NAMES = [
    'catalytic_activity',
    'domain_motif',
    'general_function',
    'protein_function'
]

ZERO_SHOT_PROMPT_ENHANCEMENT = {
    'catalytic_activity': r"""Begin the answer with the following sentence "An analysis of the protein sequence reveals that the enzyme's catalytic function corresponds to the chemical reaction: ".""",
    'domain_motif': r"""Begin the answer with the following sentence "Based on computational analysis, the provided sequence potentially contains the following protein domains or motifs: ".""",
    'general_function': r"""Begin the answer with the following sentence "A short report on the protein with the given amino acid sequence highlights: ".""",
    'protein_function': r"""Begin the answer with the following sentence "Based on the analysis of the given protein sequence, it appears that the primary function of this protein is".""",
}

DATASET_TYPES = ['training', 'validation', 'test']


def evaluate_molinst(
    model,
    task_name: str,
    dataset_dir: str,
    pdb_dir: str,
    dataset_type: str,
    save_dir: str,
    batch_size: int = 1,
    save_result: bool = True,
    use_structure: bool = True,
    use_sequence: bool = True,
    is_zero_shot: bool = False
):
    filename = f'molinst_{task_name}_{dataset_type}'
    eval_dataset = MultimodalDataset(f'{dataset_dir}/{filename}.json', pdb_dir)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    references = []
    predictions = []
    for i, eval_data in enumerate(tqdm(eval_dataloader, desc=task_name)):
        pdb_files = eval_data['pdb_files']
        sequences = eval_data['sequences']
        if not use_structure:
            pdb_files = None
        if not use_sequence:
            sequences = None
        prompts = [prompt + '\n' + ZERO_SHOT_PROMPT_ENHANCEMENT[task_name] for prompt in eval_data['prompts']] if is_zero_shot else eval_data['prompts']
        pred_response = infer(model, pdb_files, sequences, prompts)
        predictions.extend([res.strip() for res in pred_response])
        references.extend([res.strip() for res in eval_data['responses']])

    if save_result:
        json_results = []
        idx = 1
        for pred, ref in zip(predictions, references):
            json_results.append({
                'id': idx,
                'prediction': pred,
                'reference': ref
            })
            idx += 1
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(f'{save_dir}/{filename}.json', 'w') as f:
            json.dump(json_results, f, indent=4)

    rouge = load('rouge')
    rouge_score = rouge.compute(predictions=predictions, references=references)
    return rouge_score


def evaluate(
    task_name: str,
    dataset_dir: str,
    pdb_dir: str,
    dataset_type: str,
    save_dir: str,
    save_result: bool = True,
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
        protein_max_length,
    )
    evo_llama.eval()

    if task_name == 'all':
        for task_name in TASK_NAMES:
            performance = evaluate_molinst(
                evo_llama, task_name, dataset_dir, pdb_dir, dataset_type, save_dir, batch_size, save_result, use_structure, use_sequence, is_zero_shot
            )
            print(f'Task {task_name}: {performance}')
    else:
        performance = evaluate_molinst(
            evo_llama, task_name, dataset_dir, pdb_dir, dataset_type, save_dir, batch_size, save_result, use_structure, use_sequence, is_zero_shot
        )
        print(f'Task {task_name}: {performance}')


if __name__ == '__main__':
    fire.Fire(evaluate)
