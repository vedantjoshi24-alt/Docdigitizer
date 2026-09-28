"""
Fine-tunes LayoutLMv3 for token classification (question/answer/header
tagging) on the FUNSD dataset -- the benchmark referenced in the
project's literature review for form-understanding.

Run:
    python train_layoutlm.py --epochs 10 --batch_size 2 --output_dir ./layoutlmv3-funsd

This is retained as a research experiment only. The production application
does not load this checkpoint: it preserves the document as an ordered,
editable transcript instead of forcing every document into FUNSD field types.

This trains a field-tagging model (is this word part of a label or a value?).
It is separate from OCR: LayoutLMv3 consumes FUNSD's ground-truth words and
boxes, not the application's OCR output.
"""
import argparse
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
)

LABEL_LIST = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}


def build_datasets(processor):
    dataset = load_dataset("nielsr/funsd-layoutlmv3")

    def prepare(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        encoding = processor(
            images,
            batch["tokens"],
            boxes=batch["bboxes"],
            word_labels=batch["ner_tags"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        return encoding

    train_ds = dataset["train"].map(prepare, batched=True, remove_columns=dataset["train"].column_names)
    test_ds = dataset["test"].map(prepare, batched=True, remove_columns=dataset["test"].column_names)
    train_ds.set_format("torch")
    test_ds.set_format("torch")
    return train_ds, test_ds


def build_compute_metrics():
    metric = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [LABEL_LIST[p] for (p, l) in zip(pred, lab) if l != -100]
            for pred, lab in zip(predictions, labels)
        ]
        true_labels = [
            [LABEL_LIST[l] for (p, l) in zip(pred, lab) if l != -100]
            for pred, lab in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="./layoutlmv3-funsd")
    args = parser.parse_args()

    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds, test_ds = build_datasets(processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=-1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=processor,
        compute_metrics=build_compute_metrics(),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
