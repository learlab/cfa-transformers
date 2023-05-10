import os

import wandb

from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error

import torch
from torch import nn
assert torch.cuda.is_available(), 'GPU not found. You should fix this.'

from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

from datasets import DatasetDict

### Usage ###
'''
Install wandb and login to aialoe team
Run a command like the following with the argument flags that are defined in the main function:
'''

### Helper Functions ###

def load_datadict(path, tokenizer, item_id):
    ''' Loads the dataset from disk and tokenizes the text.
    '''
    def tokenize_inputs(example):
        return tokenizer(example['text'], truncation=False)
    
    return (
        DatasetDict
        .load_from_disk(os.path.join(path, item_id))
        .map(tokenize_inputs)
    )


def load_nested_datadict(path, tokenizer):
    ''' Loads a nested dataset from disk and tokenizes the text.
    This is not currently used, but could be implemented in the future.
    Could be useful for training multiple models with one command.
    '''
    def tokenize_inputs(example):
        return tokenizer(example['text'], truncation=False)
    
    dd = DatasetDict()

    subfolders = [(f.name, f.path) for f in os.scandir(path) if f.is_dir()]
    
    for item_id, subfolder in subfolders:
        dd[item_id] = (
            DatasetDict
            .load_from_disk(subfolder)
            .map(tokenize_inputs)
        )
    
    return dd


def create_model_config(model_name):
    ''' Creates a model configuration object with one label,
    which frames this as a regression-type task.
    '''
    model_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=1,
    )

    return model_config


def create_model_init(model_name, model_config):
    ''' This ensures that the weights are reloaded from the Huggingface model
    checkpoint across different runs.
    '''
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=model_config,
        ).cuda()
    
    return model_init


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    mse = mean_squared_error(labels, logits)
    rmse = mean_squared_error(labels, logits, squared=False)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def train():
    ''' The main training loop.
    '''
    wandb.init()

    config = wandb.config
    
    # set training arguments
    training_args = TrainingArguments(
        output_dir='/home/jovyan/active-projects/deidentification/bin/',
        report_to='wandb',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        optim='adamw_torch',
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_strategy='epoch',
        load_best_model_at_end=True,
        disable_tqdm=False,
    )

    # define training loop
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=datadict['train'],
        eval_dataset=datadict['dev'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # start training loop
    trainer.train()


### Main function ###

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-i',
        '--item_id',
        type=str,
        default=None,
        help='Item_id or accession')
    parser.add_argument(
        '-p',
        '--project_name',
        type=str,
        default='math-autoscore',
        help='Main project name')
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=8,
        help='batch_size')    
    parser.add_argument(
        '-s',
        '--sweep_id',
        type=str,
        default=None,
        help='sweep_id for an existing wandb sweep')
    parser.add_argument(
        '-q',
        '--dry_run',
        action='store_true',
        help='Dry run (do not log to wandb)')  

    args = parser.parse_args()
    
    # easier testing--don't log to wandb if dry run is set
    if args.dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'
    os.environ['WANDB_PROJECT'] = args.project_name
    os.environ['WANDB_LOG_MODEL'] = 'end'
    os.environ['WANDB_DIR'] = '/home/jovyan/active-projects/ellipse-methods-showcase/bin/wandb'

    model_name = 'microsoft/deberta-v3-small'
    model_config = create_model_config(model_name)
    
    model_init = create_model_init(model_name, model_config)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    data_path = '../data/naep_data.hf'
    datadict = load_datadict(data_path, tokenizer, args.item_id)
    
    if not args.sweep_id:
        # method
        sweep_config = {
            'method': 'bayes',
            'metric': {
                'name': 'eval/rmse',
                'goal': 'minimize'
            },
            'parameters':
            {
                'epochs': {
                    'values': [2, 3, 4, 5]
                },
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-6,
                    'max': 5e-5,
                },
                'dropout': {
                    'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                },                
                'weight_decay': {
                    'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                },
            },
        }

        sweep_id = wandb.sweep(sweep_config,
                               project=args.project_name)
        
    else:
        sweep_id = args.sweep_id
        
    wandb.agent(sweep_id, train, count=20)
