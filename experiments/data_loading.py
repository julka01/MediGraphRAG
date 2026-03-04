"""Data Loading Utilities."""

import os
import json
import hashlib
import datasets


def load_ds(dataset_name, seed, add_options=None):
    """Load dataset directly from HuggingFace (no external files required)."""
    user = os.environ['USER']

    train_dataset, validation_dataset = None, None
    if dataset_name == "squad":
        dataset = datasets.load_dataset("squad_v2")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    elif dataset_name == 'svamp':
        dataset = datasets.load_dataset('ChilleD/SVAMP')
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        reformat = lambda x: {
            'question': x['Question'], 'context': x['Body'], 'type': x['Type'],
            'equation': x['Equation'], 'id': x['ID'],
            'answers': {'text': [str(x['Answer'])]}}

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == 'nq':
        dataset = datasets.load_dataset("nq_open")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        reformat = lambda x: {
            'question': x['question']+'?',
            'answers': {'text': x['answer']},
            'context': '',
            'id': md5hash(str(x['question'])),
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "trivia_qa":
        dataset = datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    else:
        available_datasets = ["squad", "svamp", "nq", "trivia_qa"]
        raise ValueError(f"Dataset '{dataset_name}' not available. Choose from: {available_datasets}")

    return train_dataset, validation_dataset
