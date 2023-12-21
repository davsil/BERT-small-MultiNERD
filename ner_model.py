"""
ner_model.py

This script generates a NER model for the selected system type (A or B) for a number of epochs (defaulting to 1).
The script writes out metrics in a .json file and confusion matrix in a .csv file using a result file basename,
which defaults to "result".

usage: python ner_model.py -[hAB] -e <epochs> -r <result file basename>

"""

import sys, getopt
import numpy as np
import json
from datasets import load_dataset, load_metric
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import evaluate
import pandas as pd
from sklearn.metrics import confusion_matrix


pretrained_model_name= "prajjwal1/bert-small"
dataset_name = "Babelscape/multinerd"
new_model_name = "bert-small-multinerd"

# for system A
label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-AMIN', 'I-AMIN', 'B-BIO', 'I-BIO',
              'B-CEL', 'I-CEL', 'B-DIS', 'I-DIS', 'B-EVE', 'I-EVE', 'B-FOOD', 'I-FOOD', 'B-INST', 'I-INST',
              'B-MEDIA', 'I-MEDIA', 'B-PLANT', 'I-PLANT', 'B-MYTH', 'I-MYTH', 'B-TIME', 'I-TIME', 'B-VEHI', 'I-VEHI']

labels_vocab = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-AMIN': 7, 'I-AMIN': 8,
                'B-BIO': 9, 'I-BIO': 10, 'B-CEL': 11, 'I-CEL': 12, 'B-DIS': 13, 'I-DIS': 14, 'B-EVE': 15, 'I-EVE': 16,
                'B-FOOD': 17, 'I-FOOD': 18, 'B-INST': 19, 'I-INST': 20, 'B-MEDIA': 21, 'I-MEDIA': 22, 'B-PLANT': 23,
                'I-PLANT': 24, 'B-MYTH': 25, 'I-MYTH': 26, 'B-TIME': 27, 'I-TIME': 28, 'B-VEHI': 29, 'I-VEHI': 30}

labels_vocab_reverse = {v: k for k, v in labels_vocab.items()}

# for system B
label_list_subset = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-AMIN', 'I-AMIN', 'B-DIS', 'I-DIS']
ner_codes_subset = [labels_vocab[label] for label in label_list_subset]


def load_datasets(system_type):

    datasets = load_dataset(dataset_name)
    train_set = datasets["train"].filter(lambda x: x['lang'] == 'en')
    validate_set = datasets["validation"].filter(lambda x: x['lang'] == 'en')
    test_set = datasets["test"].filter(lambda x: x['lang'] == 'en')

    if system_type == 'B':
        def reduce_ner_classes(example):
            example['ner_tags'] = list(map(lambda x: x if x in ner_codes_subset else 0, example['ner_tags']))
            return example

        train_set = train_set.map(reduce_ner_classes)
        validate_set = validate_set.map(reduce_ner_classes)
        test_set = test_set.map(reduce_ner_classes)

    return train_set, validate_set, test_set


def tokenize(pretrained_model_name, train_set, validate_set, test_set):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    label_all_tokens = False

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], max_length=1024, truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. The label is set to -100 to be ignored in the loss function
                if word_idx is None:
                    label_ids.append(-100)
                # The label is set for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, the label is set to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    train_tokenized = train_set.map(tokenize_and_align_labels, batched=True)
    validate_tokenized = validate_set.map(tokenize_and_align_labels, batched=True)
    test_tokenized = test_set.map(tokenize_and_align_labels, batched=True)
    return tokenizer, train_tokenized, validate_tokenized, test_tokenized


def train_model(pretrained_model_name, tokenizer, new_model_name, system_type,
                train_tokenized, validate_tokenized, test_tokenized, epochs):

    model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name, num_labels=len(label_list),
                                                            label2id=labels_vocab, id2label=labels_vocab_reverse)
    training_args = TrainingArguments(
        output_dir = f"{new_model_name}_{system_type}",
        evaluation_strategy = "epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        push_to_hub=False
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metrics = evaluate.load('seqeval')

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        result = metrics.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": result["overall_precision"],
            "recall": result["overall_recall"],
            "f1": result["overall_f1"],
            "accuracy": result["overall_accuracy"],
        }

    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=validate_tokenized,   # alternative: eval_dataset=test_tokenized
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    return trainer, metrics


def do_metrics(trainer, metrics, testset, system_type, resultfilebase):

    predictions, labels, _ = trainer.predict(testset)
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    result = metrics.compute(predictions=true_predictions, references=true_labels)

    # generate and write out confusion matrix
    flat_actual = np.concatenate(true_labels)
    flat_pred = np.concatenate(true_predictions)
    if system_type == 'B':
        conf_matrix = confusion_matrix(flat_actual, flat_pred, labels=label_list_subset)
        df_conf_matrix = pd.DataFrame(conf_matrix, columns=label_list_subset, index=label_list_subset)
    else:
        conf_matrix = confusion_matrix(flat_actual, flat_pred, labels=label_list)
        df_conf_matrix = pd.DataFrame(conf_matrix, columns=label_list, index=label_list)

    df_conf_matrix.to_csv(f"{resultfilebase}.csv")

    # recast number value to int to prevent TypeError: Object of type int64 is not JSON serializable
    for key, value in result.items():
        if key not in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']:
            value['number'] = int(value['number'])

    with open(f"{resultfilebase}.json", "w") as outfile:
        json.dump(result, outfile)


def main(argv):

    help_message = 'usage: python ner_model.py -[hAB] -e <epochs> -r <resultfilebase>'
    resultfilebase = 'result'
    system_type = 'A'
    epochs = 1

    try:
        opts, args = getopt.getopt(argv, "hABe:r:")

        for opt, arg in opts:
            if opt == '-h':
                print(help_message)
                sys.exit()
            elif opt == '-A':
                system_type = 'A'
            elif opt == '-B':
                system_type = 'B'
            elif opt == '-e':
                epochs = int(arg)
            elif opt == '-r':
                resultfilebase = arg

    except getopt.GetoptError:
        sys.exit(help_message)

    # extract English-only datasets with NER tags based on system A or B 
    train_set, validate_set, test_set = load_datasets(system_type)
    # tokenize the datasets
    tokenizer, train_tokenized, validate_tokenized, test_tokenized = tokenize(pretrained_model_name,
                                                                              train_set, validate_set, test_set)
    # train (fine-tune) the model 
    trainer, metrics = train_model(pretrained_model_name, tokenizer, new_model_name, system_type,
                                   train_tokenized, validate_tokenized, test_tokenized, epochs)

    # calculate metrics test set and write to json results file
    do_metrics(trainer, metrics, test_tokenized, system_type, resultfilebase)


if __name__ == "__main__":
    main(sys.argv[1:])

