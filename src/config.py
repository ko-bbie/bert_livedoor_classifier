import transformers

MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
TEST_BATCH_SIZE = 8
EPOCHS = 10
BERT_PATH = r"C:\Users\flums\Documents\Python\pretrained_models\bert-base-japanese"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../input/train.csv"
TEST_FILE = "../input/test.csv"
TOKENIZER = transformers.AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
