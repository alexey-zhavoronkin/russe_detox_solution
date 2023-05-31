import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import trainer
from data import utils, dataset
from config import config
import pandas as pd
    
if __name__ == "__main__":

    trainer_gen = trainer.Trainer(config)
    trainer_gen.to(config['device'])

    df_train = pd.read_csv(config['df_train_path'], sep='\t')
    df_train = df_train.fillna('')
    df_train = utils.prepare_data(df_train)
    train_dataset = dataset.CustomDataset(df_train)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], drop_last=False, shuffle=False)

    df_valid = pd.read_csv(config['df_valid_path'], sep='\t')
    df_valid = df_valid.fillna('')
    df_valid = utils.prepare_data(df_valid)
    valid_dataset = dataset.CustomDataset(df_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size'], drop_last=False, shuffle=False)
    
    df_test = pd.read_csv(config['df_test_path'], sep='\t')
    df_test['neutral_comment'] = ''
    df_test = df_test.reset_index().rename(columns={'index': 'comment_id'})
    test_dataset = dataset.CustomDataset(df_test)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], drop_last=False, shuffle=False)


    if config['try_one_batch']:
        train_dataloader = [list(train_dataloader)[0]]
        valid_dataloader = [list(train_dataloader)[0]]

    trainer_gen.train(train_dataloader, valid_dataloader, test_dataloader)