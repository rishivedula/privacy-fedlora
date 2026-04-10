"""Dataset loading and preprocessing for QA tasks."""

from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer


def load_squad(
    split: str = "train",
    num_samples: Optional[int] = None
) -> Dataset:
    """Load SQuAD 2.0 dataset.

    Args:
        split: Dataset split (train, validation)
        num_samples: Optional limit on number of samples

    Returns:
        HuggingFace Dataset
    """
    dataset = load_dataset("squad_v2", split=split)

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    return dataset


def load_natural_questions(
    split: str = "train",
    num_samples: Optional[int] = None
) -> Dataset:
    """Load Natural Questions dataset.

    Uses streaming to avoid downloading full dataset (~55GB).

    Args:
        split: Dataset split (train, validation)
        num_samples: Optional limit on number of samples

    Returns:
        HuggingFace Dataset
    """
    # Use streaming to avoid downloading 55GB+ dataset
    if num_samples is not None:
        dataset = load_dataset(
            "natural_questions", "default",
            split=f"{split}[:{num_samples}]",
            trust_remote_code=True
        )
    else:
        dataset = load_dataset("natural_questions", "default", split=split, trust_remote_code=True)

    return dataset


def load_triviaqa(
    split: str = "train",
    num_samples: Optional[int] = None
) -> Dataset:
    """Load TriviaQA dataset (~2.5GB).

    Args:
        split: Dataset split (train, validation)
        num_samples: Optional limit on number of samples

    Returns:
        HuggingFace Dataset
    """
    # Use rc.nocontext for smaller download (questions + answers only)
    dataset = load_dataset("trivia_qa", "rc.nocontext", split=split)

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    return dataset


def format_triviaqa_example(example: Dict) -> Dict:
    """Format a TriviaQA example for training.

    Args:
        example: Raw TriviaQA example

    Returns:
        Formatted example with prompt and answer
    """
    question = example["question"]
    # TriviaQA rc.nocontext doesn't have context, use question directly
    context = "Answer the following trivia question."

    # Get answer - TriviaQA has multiple aliases
    answer = example["answer"]["value"]

    prompt = format_qa_prompt(question, context)

    return {
        "prompt": prompt,
        "answer": answer,
        "full_text": f"{prompt} {answer}"
    }


def load_sciq(
    split: str = "train",
    num_samples: Optional[int] = None
) -> Dataset:
    """Load SciQ dataset (Science exam questions).

    Args:
        split: Dataset split (train, validation, test)
        num_samples: Optional limit on number of samples

    Returns:
        HuggingFace Dataset
    """
    dataset = load_dataset("allenai/sciq", split=split)

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    return dataset


def load_pubmedqa(split: str = "train", num_samples: Optional[int] = None) -> Dataset:
    """Load PubMedQA biomedical QA dataset."""
    ds_split = "train" if split == "train" else "test"
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split=ds_split, trust_remote_code=True)
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset


def format_pubmedqa_example(example: Dict) -> Dict:
    question = example["question"]
    contexts = example["context"]["contexts"]
    context = " ".join(contexts)[:1000] if isinstance(contexts, list) else str(contexts)[:1000]
    answer = example.get("long_answer", example.get("final_decision", "yes"))
    prompt = format_qa_prompt(question, context)
    return {"prompt": prompt, "answer": str(answer), "full_text": f"{prompt} {answer}"}


def load_medqa(split: str = "train", num_samples: Optional[int] = None) -> Dataset:
    """Load MedQA USMLE medical licensing exam QA."""
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=split, trust_remote_code=True)
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset


def format_medqa_example(example: Dict) -> Dict:
    question = example["question"]
    options = example.get("options", {})
    context = " ".join([f"{k}: {v}" for k, v in options.items()]) if options else ""
    answer = str(example.get("answer", example.get("answer_idx", "")))
    prompt = format_qa_prompt(question, context)
    return {"prompt": prompt, "answer": answer, "full_text": f"{prompt} {answer}"}


def load_finqa(split: str = "train", num_samples: Optional[int] = None) -> Dataset:
    """Load FinQA financial QA dataset."""
    dataset = load_dataset("ibm/finqa", split=split, trust_remote_code=True)
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset


def format_finqa_example(example: Dict) -> Dict:
    question = example.get("question", "")
    pre = " ".join(example.get("pre_text", []))
    post = " ".join(example.get("post_text", []))
    context = f"{pre} {post}".strip()[:1000]
    answer = str(example.get("answer", ""))
    prompt = format_qa_prompt(question, context)
    return {"prompt": prompt, "answer": answer, "full_text": f"{prompt} {answer}"}


def load_code_qa(split: str = "train", num_samples: Optional[int] = None) -> Dataset:
    """Load CodeXGLUE code-to-text (Python) as a code-understanding QA proxy."""
    dataset = load_dataset("code_x_glue_ct_code_to_text", "python", split=split, trust_remote_code=True)
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset


def format_code_qa_example(example: Dict) -> Dict:
    code = example.get("code", "")[:500]
    docstring = example.get("docstring", "")[:200]
    question = "What does this code do?"
    prompt = format_qa_prompt(question, f"```python\n{code}\n```")
    return {"prompt": prompt, "answer": docstring, "full_text": f"{prompt} {docstring}"}


def load_arc(split: str = "train", num_samples: Optional[int] = None) -> Dataset:
    """Load AI2 ARC-Challenge science reasoning dataset."""
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset


def format_arc_example(example: Dict) -> Dict:
    question = example["question"]
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    context = " ".join([f"{l}: {t}" for l, t in zip(labels, texts)])
    answer_key = example["answerKey"]
    answer = next((t for l, t in zip(labels, texts) if l == answer_key), "")
    prompt = format_qa_prompt(question, context)
    return {"prompt": prompt, "answer": answer, "full_text": f"{prompt} {answer}"}


def load_openbookqa(split: str = "train", num_samples: Optional[int] = None) -> Dataset:
    """Load OpenBookQA science questions dataset."""
    dataset = load_dataset("allenai/openbookqa", split=split)
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset


def format_openbookqa_example(example: Dict) -> Dict:
    question = example["question_stem"]
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    context = " ".join([f"{l}: {t}" for l, t in zip(labels, texts)])
    answer_key = example["answerKey"]
    answer = next((t for l, t in zip(labels, texts) if l == answer_key), "")
    prompt = format_qa_prompt(question, context)
    return {"prompt": prompt, "answer": answer, "full_text": f"{prompt} {answer}"}


def load_commonsense_qa(split: str = "train", num_samples: Optional[int] = None) -> Dataset:
    """Load CommonsenseQA dataset."""
    dataset = load_dataset("tau/commonsense_qa", split=split)
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset


def format_commonsense_qa_example(example: Dict) -> Dict:
    question = example["question"]
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    context = " ".join([f"{l}: {t}" for l, t in zip(labels, texts)])
    answer_key = example["answerKey"]
    answer = next((t for l, t in zip(labels, texts) if l == answer_key), "")
    prompt = format_qa_prompt(question, context)
    return {"prompt": prompt, "answer": answer, "full_text": f"{prompt} {answer}"}


def format_sciq_example(example: Dict) -> Dict:
    """Format a SciQ example for training.

    Args:
        example: Raw SciQ example

    Returns:
        Formatted example with prompt and answer
    """
    question = example["question"]
    # SciQ has support text as context
    context = example.get("support", "")
    if not context:
        context = "No additional context provided."

    answer = example["correct_answer"]

    prompt = format_qa_prompt(question, context)

    return {
        "prompt": prompt,
        "answer": answer,
        "full_text": f"{prompt} {answer}"
    }


def format_qa_prompt(question: str, context: str) -> str:
    """Format question and context into instruction prompt.

    Args:
        question: The question to answer
        context: The context/passage containing the answer

    Returns:
        Formatted prompt string
    """
    return f"""Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:"""


def format_squad_example(example: Dict) -> Dict:
    """Format a SQuAD example for training.

    Args:
        example: Raw SQuAD example

    Returns:
        Formatted example with prompt and answer
    """
    question = example["question"]
    context = example["context"]

    # SQuAD 2.0 may have empty answers (unanswerable)
    answers = example["answers"]["text"]
    answer = answers[0] if answers else "unanswerable"

    prompt = format_qa_prompt(question, context)

    return {
        "prompt": prompt,
        "answer": answer,
        "full_text": f"{prompt} {answer}"
    }


def format_nq_example(example: Dict) -> Dict:
    """Format a Natural Questions example for training.

    Args:
        example: Raw NQ example

    Returns:
        Formatted example with prompt and answer
    """
    question = example["question"]["text"]

    # Get document text (simplified - NQ has complex structure)
    doc_tokens = example["document"]["tokens"]
    doc_text = " ".join([t["token"] for t in doc_tokens[:500]])  # Truncate

    # Get short answer if available
    annotations = example["annotations"]
    answer = "unanswerable"
    if annotations and annotations[0]["short_answers"]:
        sa = annotations[0]["short_answers"][0]
        start, end = sa["start_token"], sa["end_token"]
        answer_tokens = [doc_tokens[i]["token"] for i in range(start, min(end, len(doc_tokens)))]
        answer = " ".join(answer_tokens)

    prompt = format_qa_prompt(question, doc_text)

    return {
        "prompt": prompt,
        "answer": answer,
        "full_text": f"{prompt} {answer}"
    }


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    dataset_type: str,
    max_length: int = 512
) -> Dataset:
    """Preprocess and tokenize dataset.

    Args:
        dataset: Raw dataset
        tokenizer: Tokenizer for encoding
        dataset_type: "squad", "nq", or "sciq"
        max_length: Maximum sequence length

    Returns:
        Tokenized dataset
    """
    format_fns = {
        "squad": format_squad_example,
        "nq": format_nq_example,
        "triviaqa": format_triviaqa_example,
        "sciq": format_sciq_example,
        "pubmedqa": format_pubmedqa_example,
        "medqa": format_medqa_example,
        "finqa": format_finqa_example,
        "code_qa": format_code_qa_example,
        "arc": format_arc_example,
        "openbookqa": format_openbookqa_example,
        "commonsense_qa": format_commonsense_qa_example,
    }
    format_fn = format_fns.get(dataset_type)
    if format_fn is None:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    def tokenize(example):
        formatted = format_fn(example)
        encoded = tokenizer(
            formatted["full_text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """Create DataLoader from dataset.

    Args:
        dataset: Tokenized dataset
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        PyTorch DataLoader
    """
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_client_data(
    client_id: str,
    config: Dict,
    tokenizer: PreTrainedTokenizer
) -> Tuple[Dataset, Dataset]:
    """Load and preprocess data for a specific client.

    Args:
        client_id: Client identifier (c1, c2, etc.)
        config: Configuration dict with client data specs
        tokenizer: Tokenizer for preprocessing

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    client_config = config["clients"][client_id]
    dataset_name = client_config["dataset"]
    num_samples = client_config.get("num_samples", 10000)

    # Registry: dataset_name -> (loader_fn, eval_split, dataset_type)
    registry = {
        "squad_v2":       (load_squad,          "validation", "squad"),
        "triviaqa":       (load_triviaqa,        "validation", "triviaqa"),
        "sciq":           (load_sciq,            "validation", "sciq"),
        "pubmedqa":       (load_pubmedqa,        "test",       "pubmedqa"),
        "medqa":          (load_medqa,           "test",       "medqa"),
        "finqa":          (load_finqa,           "test",       "finqa"),
        "code_qa":        (load_code_qa,         "test",       "code_qa"),
        "arc":            (load_arc,             "validation", "arc"),
        "openbookqa":     (load_openbookqa,      "validation", "openbookqa"),
        "commonsense_qa": (load_commonsense_qa,  "validation", "commonsense_qa"),
    }
    if dataset_name not in registry:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    load_fn, eval_split, dataset_type = registry[dataset_name]
    train_data = load_fn("train", num_samples)
    eval_data = load_fn(eval_split, min(1000, num_samples // 10))

    max_length = config["training"].get("max_seq_length", 512)

    train_dataset = preprocess_dataset(train_data, tokenizer, dataset_type, max_length)
    eval_dataset = preprocess_dataset(eval_data, tokenizer, dataset_type, max_length)

    return train_dataset, eval_dataset
