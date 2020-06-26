import sentencepiece as sp


class BertSentencePieceTokenizer:
    def __init__(self, model_path):
        spm = sp.SentencePieceProcessor()
        spm.Load(model_path)
        self.tokenizer = spm
        self.unk_id = self.tokenizer.piece_to_id("<unk>")
        self.cls_id = self.tokenizer.piece_to_id("[CLS]")
        self.sep_id = self.tokenizer.piece_to_id("[SEP]")
        self.pad_id = self.tokenizer.piece_to_id("[PAD]")

    def encode_plus(
        self, text, text_pair=None, add_special_tokens=True, max_length=None
    ):
        if add_special_tokens:
            if text_pair:
                ids = (
                    [self.cls_id]
                    + self.tokenizer.encode_as_ids(text_pair)
                    + [self.sep_id]
                    + self.tokenizer.encode_as_ids(text)
                    + [self.sep_id]
                )
                mask = [1] * len(ids)
                token_type_ids = [0] * (
                    len(self.tokenizer.encode_as_ids(text_pair) + 2)
                ) + [1] * (len(self.tokenizer.encode_as_ids(text) + 1))
                tokens = (
                    ["[CLS]"]
                    + self.tokenizer.encode_as_pieces(text_pair)
                    + ["[SEP]"]
                    + self.tokenizer.encode_as_pieces(text)
                    + ["[SEP]"]
                )
            else:
                ids = [self.cls_id] + self.tokenizer.encode_as_ids(text) + [self.sep_id]
                mask = [1] * len(ids)
                token_type_ids = [0] * len(ids)
                tokens = ["[CLS]"] + self.tokenizer.encode_as_pieces(text) + ["[SEP]"]
        else:
            tokens = self.tokenizer.encode_as_pieces(text)
        return {
            "tokens": tokens,
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": token_type_ids,
        }


if __name__ == "__main__":
    tokenizer = BertSentencePieceTokenizer(
        r"C:\Users\flums\Documents\Python\pretrained_models\bert-yoheikikuta\wiki-ja.model"
    )
    inputs = tokenizer.encode_plus("試しに文章をencodeしてみる。", max_length=256)
    for k, v in inputs.items():
        print(f"{k}: {v}")
