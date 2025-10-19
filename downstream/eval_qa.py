import logging
import math
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
import time

import re
import string
import csv
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm

import datasets
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils import check_min_version #, send_example_telemetry
from transformers.utils.versions import require_version

from models import MistralMLPModel, MLPMemory

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.21.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacreblue) on a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the metrics (sacreblue) on a jsonlines file."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=4096,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    eval_subset: str = field(default='validation')
    patience: int = field(default=None)

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        # accepting both json and jsonl file extensions, as
        # many jsonlines files actually have a .json extension
        valid_extensions = ["json", "jsonl"]

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in valid_extensions, "`train_file` should be a jsonlines file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in valid_extensions, "`validation_file` should be a jsonlines file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

@dataclass
class KNNArguments:
    # neuralknn
    use_neural_knn: bool = field(default=False)
    knn_generator_path: str = field(default=None)
    lmbda: float = field(default=0.25)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, KNNArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, knn_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, knn_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_translation", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(42)

    # Get the datasets: you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            trust_remote_code=True,
            token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        padding_side='left',
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch.float16,
    )

    #if llama
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    model = model.eval()
    model.to("cuda")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    elif knn_args.build_index:
        logger.info("Building index")
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    # padding = "max_length" if data_args.pad_to_max_length else "max_length"
    padding = "max_length" if data_args.pad_to_max_length else False
    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets[data_args.eval_subset]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        print("eval dataset is:",eval_dataset[0])

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # ============================================= Previous Misc ===============================================

    model = model.eval()
    model.to("cuda")

    # Injecting KNN
    dimension = model.config.hidden_size
    knn_wrapper = None
    knn_args.seed = training_args.seed

    model.resize_token_embeddings(len(tokenizer))

    if knn_args.use_neural_knn and knn_args.knn_generator_path is not None:
        config = AutoConfig.from_pretrained(knn_args.knn_generator_path)
        knn_generator = MistralMLPModel.from_pretrained(
            knn_args.knn_generator_path,
            config=config,
            input_dim=config.hidden_size,
            output_dim=config.hidden_size
        ).to(model.device)

        knn_generator.eval()

        # Create MLPMemory model
        model = MLPMemory(base_lm=model, knn_generator=knn_generator, lmbda=knn_args.lmbda, knn_temp=1.0)
    # ============================================= Start Evaluation ===============================================

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    def normalize_question(question):
        if not question.endswith("?"):
            question = question + "?"
        return question[0].lower() + question[1:]

    def build_qa_prompt(example, fewshot, dataset_path):
        if fewshot:
            few_shot_examples = [
                "Question: Which magazine was started first Arthur's Magazine or First for Women?\nAnswer: Arthur's Magazine\n",
                "Question: Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?\nAnswer: Jonathan Stark\n",
                "Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?\nAnswer: no\n",
                "Question: Where is the base of one of the three main British intelligence agencies?\nAnswer: Cheltenham\n",
                "Question: Gary Harrison, began his career in the 1970s and has written over how many major-label recorded songs including several number-one hits, another artist who have recorded his work include Bryan White, an American country music artist?\nAnswer: 300\n",
            ]
            combined_examples = "".join(few_shot_examples)
        else:
            combined_examples = ""
        question_text = normalize_question(example["question"])
        ex_prompt = f"Answer these questions:\n{combined_examples}Question: {question_text}\nAnswer:"
        return ex_prompt

    def normalize_answer(s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


    def text_has_answer(answers, text) -> bool:
        text = normalize_answer(text)
        for single_answer in answers:
            single_answer = normalize_answer(single_answer)
            if single_answer in text:
                return True
        return False


    def exact_match(prediction, ground_truth):
        return normalize_answer(prediction) == normalize_answer(ground_truth)

    def f1_score(prediction, ground_truth):
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        ZERO_METRIC = (0, 0, 0)
        ONE_METRIC = (1, 1, 1)

        if normalized_ground_truth in ['yes', 'no', 'noanswer']:
            if normalized_ground_truth in normalized_prediction:
                return ONE_METRIC
            else:
                return ZERO_METRIC

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return ZERO_METRIC
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1, precision, recall

    def get_answer_from_model_output(outputs, tokenizer, prompt):
        generation_str = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
        generation_str = generation_str[len(prompt):]
        answer = generation_str.split("\n")[0]
        return answer, generation_str

    def compute_loss(tokenizer, model, device, prompt, answer, max_length=4096):
        text = prompt + answer
        with torch.no_grad():
            text_encoded = tokenizer.encode(text, return_tensors="pt").to(device)
            num_tokens_to_truncate = 0
            if text_encoded.size(1) > max_length:
                num_tokens_to_truncate = text_encoded.size(1) - max_length
                text_encoded = text_encoded[:, num_tokens_to_truncate:]
            ctx = tokenizer.encode(prompt, return_tensors="pt")
            ctx = ctx.view(-1).tolist()
            idx = len(ctx) - 1 - num_tokens_to_truncate
            outputs = model(text_encoded, labels=text_encoded)
            loss = outputs.loss

        eval_loss = loss[idx:]
        eval_loss = eval_loss.mean().item()

        del text_encoded
        torch.cuda.empty_cache()
        return eval_loss

    def process_answers(ex, dataset_path):
        if "hotpot" in dataset_path:
            answers = ex["answer"]
            if isinstance(answers, str):
                answers = [answers]
        elif "nq" in dataset_path:
            answers = ex["answer"]
        return answers

    def preprocess_dataset(dataset):
        eval_dataset = []
        for ex in dataset:
            answers = set(ex["answer"])
            answers = list(answers)
            #remove wrong cases
            flag =1
            for answer in answers:
                if  answer==")":
                    flag = 0
            if flag:
                eval_dataset.append(ex)
        return eval_dataset

    def evaluate_dataset(
            model, tokenizer, device, eval_dataset, dataset_path, max_length, output_dir=None, max_tokens_to_generate=15):
        idx = 0
        num_correct = 0
        num_has_answer = 0
        num_too_long = 0
        sample_prompt = None
        pred_losses = []
        f1_list = []
        prec_list = []
        recall_list = []
        results = []

        for ex in (tq := tqdm(eval_dataset, desc=f"F1:  0.0%")):

            answers = process_answers(ex, dataset_path)
            losses = []
            prompt = build_qa_prompt(ex, False, dataset_path)

            if idx == 0:
                sample_prompt = prompt
            has_answer = text_has_answer(answers, prompt)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            if input_ids.shape[-1] > max_length - max_tokens_to_generate:
                num_too_long += 1
                input_ids = input_ids[..., -(max_length - max_tokens_to_generate):]

            with torch.no_grad():
                outputs = model.generate(input_ids,  pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_tokens_to_generate)

            prompt_length = tokenizer(prompt,return_length=True).length

            prediction, generation = get_answer_from_model_output(outputs, tokenizer, prompt)

            is_correct = any([exact_match(prediction, answer) for answer in answers])
            max_f1 = 0
            best_prec = 0
            best_recall = 0
            for answer in answers:
                f1, prec, recall = f1_score(prediction, answer)
                if f1 > max_f1:
                    max_f1 = f1
                    best_prec = prec
                    best_recall = recall

            f1_list.append(max_f1)
            prec_list.append(best_prec)
            recall_list.append(best_recall)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            if idx < len(eval_dataset):
                results.append({'Question':ex['question'], 'Prediction': prediction, 'Answers': answers, 'Correct':is_correct, 'F1':max_f1})
            if idx == len(eval_dataset) -1:
                with open(os.path.join(output_dir,f"predictions_and_answers_{len(eval_dataset)}.csv"), 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['Question', 'Prediction', 'Answers', 'Correct', 'F1']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writeheader()
                    for result in results:
                        result['Answers'] = str(result['Answers'])
                        writer.writerow(result)
            idx += 1
            if is_correct:
                num_correct += 1
            if has_answer:
                num_has_answer += 1
            tq.set_description(f"F1: {sum(f1_list) / idx * 100:4.1f}%")

        em = num_correct / idx * 100
        has_answer = num_has_answer / idx * 100
        F1 = sum(f1_list)/len(f1_list) * 100

        print(f"EM: {em:.2f}%")
        print(f"F1 score: {F1:.2f}%")
        print(f"% of prompts with answer: {num_has_answer / idx * 100:.1f}%")
        if output_dir is not None:
            d = {"em": em, "f1": F1, "has_answer": has_answer, "num_examples": idx, "too_long": num_too_long}
            with open(os.path.join(output_dir, "eval.json"), "w") as f:
                f.write(json.dumps(d) + "\n")
            if sample_prompt is not None:
                with open(os.path.join(output_dir, "example_prompt.txt"), "w") as f:
                    f.write(sample_prompt)

    if training_args.do_eval:
        if 'nq' in data_args.dataset_name:
            eval_dataset = preprocess_dataset(eval_dataset)

        evaluate_dataset(
            model, tokenizer, training_args.device, eval_dataset, data_args.dataset_name,
            max_length=max_length,
            output_dir=training_args.output_dir,
        )

    kwargs = {}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if knn_wrapper is not None:
        knn_wrapper.break_out()

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
