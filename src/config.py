from sp_tokenizer import BertSentencePieceTokenizer

MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
TEST_BATCH_SIZE = 8
EPOCHS = 10
BERT_PATH = r"C:\Users\flums\Documents\Python\pretrained_models\bert-yoheikikuta\model.ckpt-1400000"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../input/train.csv"
TEST_FILE = "../input/test.csv"
TOKENIZER = BertSentencePieceTokenizer(
    r"C:\Users\flums\Documents\Python\pretrained_models\bert-yoheikikuta\wiki-ja.model"
)
