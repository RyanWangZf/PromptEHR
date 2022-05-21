# save all constant variables used by the package

CODE_TYPES = [
    'diagnosis',
    'procedure',
    'drug',
]

SPECIAL_TOKEN_DICT = {
    'diagnosis': ['<diag>', '</diag>'],
    'procedure': ['<prod>', '</prod>'],
    'drug': ['<drug>', '</drug>'],
}

UNKNOWN_TOKEN = '<unk>'

model_max_length = 512
