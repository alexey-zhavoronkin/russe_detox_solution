import torch

config = dict()
config['df_train_path'] = '../data/input/train.tsv'
config['df_valid_path'] = '../data/input/dev.tsv'
config['df_test_path'] = '../data/input/test.tsv'
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
config['model_name_or_path'] = 'sberbank-ai/ruT5-large'
config['optimizer'] = 'Adafactor'
config['scheduler_step_size'] = 1000
config['gamma'] = 0.99
config['max_length'] = 50
config['max_generated_length'] = 50
config['epoch_num'] = 500
config['logs_path'] = 'training_logs'
config['try_one_batch'] = False
config['seed'] = 123
config['learning_rate'] = 1e-3
config['batch_size'] = 8
config['num_beams'] = 10
config['temperature'] = 50.0