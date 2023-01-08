# save all constant variables used by the package
CODE_TYPES = ['tbd']
SPECIAL_TOKEN_DICT = {'tbd':['<tbd>','</tbd>']}

UNKNOWN_TOKEN = '<unk>'

model_max_length = 512

eps = 1e-16

# PRETRAINED_MODEL_URL = 'https://uofi.box.com/shared/static/cu09as2bmotrr9bejsfgumv6yx46mmfw.zip'
PRETRAINED_MODEL_URL = 'https://storage.googleapis.com/pytrial/promptEHR_pretrained.zip'

SYNTHETIC_DATA_URL = 'https://github.com/RyanWangZf/PromptEHR/raw/main/demo_data/synthetic_ehr/data.pkl'

# a name mapping from the original promptehr config to the training_args
config_to_train_args = {
    'epochs': 'num_train_epochs',
    'num_worker': 'dataloader_num_workers',
    'batch_size': 'per_device_train_batch_size',
    'eval_batch_size': 'per_device_eval_batch_size',
    'eval_step': 'eval_steps',
}