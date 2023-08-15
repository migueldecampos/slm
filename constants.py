# Prediction modes
PREV_TOKEN = "previous_token"
NEXT_TOKEN = "next_token"
PREDICTION_MODES = [PREV_TOKEN, NEXT_TOKEN]

TINYSTORIES_DATASOURCE = "tinystories"
SHAKESPEARE_DATASOURCE = "shakespeare"
DATASOURCES = [TINYSTORIES_DATASOURCE, SHAKESPEARE_DATASOURCE]

# Errors
WRONG_PREDICTION_MODE = "Wrong prediction mode."
TINYSTORIES_NOT_TOKENIZED = (
    "The TinyStories hasn't been tokenized yet, tokenize before proceeding."
)


# File names
TINYSTORIES_METADATA_JSON = "tinystories_metadata.json"
TOKENIZED_TINYSTORIES_VAL = "tokenized_tinystories_val.txt"
TOKENIZED_TINYSTORIES_TRAIN = "tokenized_tinystories_train.txt"
