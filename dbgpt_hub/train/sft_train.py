import copy
import os
import sys

from dbgpt_hub.configs.data_args import DatasetAttr

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

from dbgpt_hub.llm_base.loggings import LogCallback, get_logger
from dbgpt_hub.llm_base.config_parser import get_train_args
from dbgpt_hub.llm_base.load_tokenizer import load_model_and_tokenizer
from dbgpt_hub.data_process.data_utils import (
    get_dataset,
    preprocess_dataset,
    split_dataset,
)
from dbgpt_hub.configs.config import EXT2TYPE, IGNORE_INDEX
from dbgpt_hub.llm_base.model_trainer import (
    Seq2SeqPeftTrainer,
    ComputeMetrics,
    get_logits_processor,
    plot_loss,
)

from datasets import (
    load_dataset
)

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from dbgpt_hub.configs.model_args import (
        ModelArguments,
        FinetuningArguments,
        GeneratingArguments,
    )
    from dbgpt_hub.configs.data_args import DataArguments


logger = get_logger(__name__)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
    args = None
):
    dataset = get_dataset(model_args, data_args)

    model, tokenizer = load_model_and_tokenizer(
        model_args, finetuning_args, training_args.do_train
    )

    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, "sft")
    if training_args.do_eval:

        dataset_attr = DatasetAttr(
                    "example_text2sql_dev.json",
                    dataset_name="example_text2sql_dev.json",
                    dataset_sha1=None,
                    stage=None,
                )
        
        data_path = None
        data_files: List[str] = []

        if os.path.isdir(
            os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        ):  # directory
            for file_name in os.listdir(
                os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
            ):
                data_files.append(
                    os.path.join(
                        data_args.dataset_dir, dataset_attr.dataset_name, file_name
                    )
                )
                if data_path is None:
                    data_path = EXT2TYPE.get(file_name.split(".")[-1], None)
                else:
                    assert data_path == EXT2TYPE.get(
                        file_name.split(".")[-1], None
                    ), "file type does not match."
        elif os.path.isfile(
            os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        ):  # single file
            data_files.append(
                os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
            )
            data_path = EXT2TYPE.get(dataset_attr.dataset_name.split(".")[-1], None)
        else:
            raise ValueError("File not found.")

        assert data_path, "File extension must be txt, csv, json or jsonl."

        eval_dataset = load_dataset(
            data_path,
            data_files=data_files,
            split=data_args.split,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        print(f"len train dataset: {len(dataset)}")
        print(f"len dataset: {len(eval_dataset)}")

        for column_name in ["prompt", "query", "response", "history"]:  # align datasets
            if (
                getattr(dataset_attr, column_name)
                and getattr(dataset_attr, column_name) != column_name
            ):
                eval_dataset = eval_dataset.rename_column(
                    getattr(dataset_attr, column_name), column_name
                )

        eval_data_args = copy.deepcopy(data_args)
        eval_data_args.dataset = "example_text2sql_dev.json"
        eval_data_args.dataset_list = ["example_text2sql_dev.json"]
        eval_dataset = preprocess_dataset(eval_dataset, tokenizer, eval_data_args, training_args, "sft")
    else:
        eval_dataset = dataset


    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX
        if data_args.ignore_pad_token_for_loss
        else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args_dict = training_args.to_dict()
    training_args_dict.update(
        dict(
            generation_max_length=training_args.generation_max_length
            or data_args.max_target_length,
            generation_num_beams=data_args.eval_num_beams
            or training_args.generation_num_beams,
        )
    )
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Initialize our Trainer
    trainer = Seq2SeqPeftTrainer(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer)
        if training_args.predict_with_generate
        else None,
        # **split_dataset(dataset, data_args, training_args)
        train_dataset=dataset,
        eval_dataset=eval_dataset
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = list(
        set([tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids)
    )
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Evaluation before starting training
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if (
            training_args.predict_with_generate
        ):  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if (
            training_args.predict_with_generate
        ):  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(
            dataset, metric_key_prefix="predict", **gen_kwargs
        )
        if (
            training_args.predict_with_generate
        ):  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)


def train(
    args: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    (
        model_args,
        data_args,
        training_args,
        finetuning_args,
        generating_args,
    ) = get_train_args(args)
 
    callbacks = [LogCallback()] if callbacks is None else callbacks
    
    run_sft(
        model_args,
        data_args,
        training_args,
        finetuning_args,
        generating_args,
        callbacks,
    )


def export_model(
    args: Optional[Dict[str, Any]] = None, max_shard_size: Optional[str] = "10GB"
):
    model_args, _, training_args, finetuning_args, _ = get_train_args(args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
    model.save_pretrained(training_args.output_dir, max_shard_size=max_shard_size)
    try:
        tokenizer.save_pretrained(training_args.output_dir)
    except:
        logger.warning("Cannot save tokenizer, please copy the files manually.")


if __name__ == "__main__":
    train()
