"""Fine-tune DistilBERT for extractive QA on SQuAD with the Hugging Face Trainer.

CPU default (finishes in well under an hour on a laptop):
    python -m qa.train
Internship-report setting (10k samples, 3 epochs) - use a GPU / Colab:
    python -m qa.train --train-samples 10000 --epochs 3
"""
import argparse
import json
import time

import torch
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

from . import config
from .data import load_squad, sample


def prepare_train_features(examples, tokenizer, max_length, stride):
    """Tokenise question/context pairs into overlapping windows and label answer spans.

    Windows that do not contain the answer are labelled with the [CLS] position (0).
    """
    questions = [q.strip() for q in examples["question"]]
    enc = tokenizer(
        questions, examples["context"],
        truncation="only_second", max_length=max_length, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True,  # padded per batch by the collator
    )
    sample_map = enc.pop("overflow_to_sample_mapping")
    offsets = enc.pop("offset_mapping")
    starts, ends = [], []
    for i, offset in enumerate(offsets):
        answer = examples["answers"][sample_map[i]]
        a_start = answer["answer_start"][0]
        a_end = a_start + len(answer["text"][0])
        seq_ids = enc.sequence_ids(i)
        ctx_start = seq_ids.index(1)
        ctx_end = len(seq_ids) - 1 - seq_ids[::-1].index(1)
        if offset[ctx_start][0] > a_start or offset[ctx_end][1] < a_end:
            starts.append(0)
            ends.append(0)
            continue
        s = ctx_start
        while s <= ctx_end and offset[s][0] <= a_start:
            s += 1
        e = ctx_end
        while e >= ctx_start and offset[e][1] >= a_end:
            e -= 1
        starts.append(s - 1)
        ends.append(e + 1)
    enc["start_positions"] = starts
    enc["end_positions"] = ends
    return enc


def main():
    gpu = torch.cuda.is_available()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=config.BASE_READER)
    ap.add_argument("--output-dir", default=str(config.FINETUNED_READER_DIR))
    ap.add_argument("--train-samples", type=int, default=10000 if gpu else 3000)
    ap.add_argument("--eval-samples", type=int, default=1000 if gpu else 300)
    ap.add_argument("--epochs", type=float, default=3 if gpu else 1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=config.MAX_LENGTH)
    ap.add_argument("--stride", type=int, default=config.DOC_STRIDE)
    ap.add_argument("--max-steps", type=int, default=-1, help="Override epochs with a fixed step count")
    args = ap.parse_args()

    set_seed(config.SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.base_model)

    # Training data from SQuAD train; validation loss on held-out SQuAD validation questions.
    train_raw = sample(load_squad("train"), args.train_samples)
    eval_raw = sample(load_squad("validation"), args.eval_samples, seed=config.SEED + 1)
    fn_kwargs = dict(tokenizer=tokenizer, max_length=args.max_length, stride=args.stride)
    train_ds = train_raw.map(prepare_train_features, batched=True, remove_columns=train_raw.column_names,
                             fn_kwargs=fn_kwargs, desc="Tokenising train")
    eval_ds = eval_raw.map(prepare_train_features, batched=True, remove_columns=eval_raw.column_names,
                           fn_kwargs=fn_kwargs, desc="Tokenising validation")
    print(f"train examples={len(train_raw)} features={len(train_ds)} | "
          f"eval examples={len(eval_raw)} features={len(eval_ds)} | device={'cuda' if gpu else 'cpu'}")

    targs = TrainingArguments(
        output_dir=str(config.ROOT / "models" / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        weight_decay=0.01,
        warmup_steps=0.1,  # float < 1 = fraction of total steps
        fp16=gpu,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        seed=config.SEED,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=DataCollatorWithPadding(tokenizer))

    before = trainer.evaluate()
    print(f"validation loss before fine-tuning: {before['eval_loss']:.4f}")
    t0 = time.time()
    train_out = trainer.train()
    train_seconds = time.time() - t0
    after = trainer.evaluate()
    print(f"validation loss after fine-tuning:  {after['eval_loss']:.4f}  ({train_seconds / 60:.1f} min)")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    log = {
        "base_model": args.base_model,
        "train_examples": len(train_raw),
        "train_features": len(train_ds),
        "eval_examples": len(eval_raw),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "global_steps": train_out.global_step,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "stride": args.stride,
        "device": "cuda" if gpu else "cpu",
        "cpu": config.cpu_name(),
        "train_minutes": round(train_seconds / 60, 2),
        "final_train_loss": round(train_out.training_loss, 4),
        "eval_loss_before": round(before["eval_loss"], 4),
        "eval_loss_after": round(after["eval_loss"], 4),
        "loss_history": [h for h in trainer.state.log_history if "loss" in h],
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"Saved model to {args.output_dir} and log to {config.RESULTS_DIR / 'training_log.json'}")


if __name__ == "__main__":
    main()
