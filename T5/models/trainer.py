import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime
import os

from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import Adafactor, get_polynomial_decay_schedule_with_warmup
from evaluation import evaluation
from loggers import txt_logger

class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()

        
        self.model = T5ForConditionalGeneration.from_pretrained(config['model_name_or_path'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'])
        if config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        elif config['optimizer'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['learning_rate'])
        elif config['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config['learning_rate'])
        elif config['optimizer'] == 'Adafactor':
            self.optimizer = Adafactor(self.model.parameters(), lr=config['learning_rate'], relative_step=False)
#         self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['scheduler_step_size'], gamma=config['gamma'])
        self.scheduler = get_polynomial_decay_schedule_with_warmup(self.optimizer, power=0.5, num_warmup_steps=3000, num_training_steps=30_000)
        self.epoch_num = config['epoch_num']
        self.device = config['device']
        self.max_length = config['max_length']
        self.logger = txt_logger.TXTLogger(config['logs_path'])
        self.batch_size = config['batch_size']
        self.num_beams = config['num_beams']
        self.temperature = config['temperature']
        self.seed_torch(config['seed'])
        

    def seed_torch(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def encode_batch(self, batch):
        src = self.tokenizer.batch_encode_plus(
            batch['src_text'],
            padding = 'longest',
            truncation = True,
            max_length = self.max_length,
            return_tensors="pt",
        )
        tgt = self.tokenizer.batch_encode_plus(
            batch['tgt_text'],
            padding = 'longest',
            max_length = self.max_length,
            truncation = True,
            return_tensors="pt",
        )
        batch_encd = {
            'input_ids': src['input_ids'].to(self.device), 
            'attention_mask': src['attention_mask'].to(self.device), 
            'labels': tgt['input_ids'].to(self.device),
            'decoder_attention_mask': tgt['attention_mask'].to(self.device)
        }
        return batch_encd
        
    def training_step(self, batch):
        
        batch_encd = self.encode_batch(batch)
#         print(batch_encd['input_ids'].shape)
        
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.model(**batch_encd).loss
        
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
        
    def validation_step(self, batch):
        
        batch_encd = self.encode_batch(batch)
        
        self.model.eval()
        with torch.no_grad():
            loss = self.model(**batch_encd).loss
        
        return loss.item()

    def forward(self, batch, num_beams, temperature):
        
        batch_encd = self.encode_batch(batch)
        
        generated_ids = self.model.generate(
            input_ids = batch_encd['input_ids'],
            attention_mask = batch_encd['attention_mask'], 
            max_length=30, 
            num_beams=num_beams,
            temperature=temperature,
            repetition_penalty=3.0, 
            length_penalty=1.0, 
            early_stopping=True
          )

        source = [self.tokenizer.decode(x, skip_special_tokens=True) for x in batch_encd['input_ids']]
        preds =  [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids     ]
        target = [self.tokenizer.decode(t, skip_special_tokens=True) for t in batch_encd['labels']   ]

          
        return source, preds, target
    
    def evaluate(self, df):

        original = df['source_sentences'].tolist()
        rewritten = df['predicted_sentences'].tolist()
        neutral_references = df['actual_sentences'].tolist()
        
        results = evaluation.evaluate(original, rewritten, neutral_references, self.batch_size)

        return results
    
    def predict(self, test_dataloader, num_beams=2, temperature=0.0):
        comment_ids = []
        source_sentences = []
        predicted_sentences = []
        actual_sentences = []
        for ids, batch in tqdm(test_dataloader):
            source, preds, target = self.forward(batch, num_beams, temperature)

            comment_ids.extend(map(lambda x: x.item(), ids))
            source_sentences.extend(source)
            predicted_sentences.extend(preds)
            actual_sentences.extend(target)
            
        df = pd.DataFrame({
            'comment_ids': comment_ids, 
            'source_sentences': source_sentences, 
            'predicted_sentences': predicted_sentences, 
            'actual_sentences': actual_sentences,
        })
        df = df.groupby('comment_ids').agg(
            source_sentences=('source_sentences', lambda x: x.iloc[0]),
            predicted_sentences=('predicted_sentences', lambda x: x.iloc[0]),
            actual_sentences=('actual_sentences', lambda x: list(x))
        )
        results = self.evaluate(df)
        
        filename = f"./checkpoints/df_{results['joint']:.5f}.txt"
        df.to_csv(
            path_or_buf=filename, 
            columns=['predicted_sentences'], 
            index=False, 
            header=False
        )
        df.to_csv(
            path_or_buf=f"../data/df_output.txt", 
            index=False, 
            header=False
        )
        print(df.sample(10).iloc[:, :2])
        
        files_to_remove = sorted(os.listdir('./checkpoints/'))[:-5]
        [os.remove('./checkpoints/'+file) for file in files_to_remove]
        
        return results
        

    def train(self, train_dataloader, val_dataloader, test_dataloader):
        try:
            for epoch in tqdm(range(self.epoch_num)):
                train_epoch_loss = 0
                for ids, batch in tqdm(train_dataloader):
                    train_loss = self.training_step(batch)
                    train_epoch_loss += train_loss
                train_epoch_loss = train_epoch_loss / len(train_dataloader)
                
                val_epoch_loss = 0
                if val_dataloader is not None:
                    for ids, batch in tqdm(val_dataloader):
                        val_loss = self.validation_step(batch)
                        val_epoch_loss += val_loss
                    val_epoch_loss = val_epoch_loss / len(val_dataloader)

                results = self.predict(test_dataloader, self.num_beams, self.temperature)
                
                results = [
                    val_epoch_loss,
                    train_epoch_loss,
                    results['accuracy'],
                    results['similarity'],
                    results['fluency'],
                    results['joint'],
                    results['chrf']
                ]
                
                print(f'|{"VAL_LOSS":^10}|{"TRAIN_LOSS":^10}|{"ACC":^10}|{"SIM":^10}|{"FL":^10}|{"J":^10}|{"ChrF1":^10}|')
                print('|' + '|'.join(['-' * 10 for _ in range(len(results))]) + '|')
                print('|' + '|'.join([f'{x:^10.4f}' for x in results]) + '|')
                self.logger.log({
                    'val_loss': results[0],
                    'train_loss': results[1],
                    'accuracy': results[2],
                    'similarity': results[3],
                    'fluency': results[4],
                    'joint': results[5],
                    'chrf': results[6],
                })

        except KeyboardInterrupt:
            pass

        print(f"Last {epoch} epoch train loss: {results[0]}")
        print(f"Last {epoch} epoch val loss: {results[1]}")
        print(f"Last {epoch} epoch val accuracy: {results[2]}")
        print(f"Last {epoch} epoch val similarity: {results[3]}")
        print(f"Last {epoch} epoch val fluency: {results[4]}")
        print(f"Last {epoch} epoch val joint: {results[5]}")
        print(f"Last {epoch} epoch val chrf: {results[6]}")
        
        

