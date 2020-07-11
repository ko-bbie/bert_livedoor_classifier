import pandas as pd
import torch

import config
import utils


class LivedoorDataset:
    def __init__(self, article, targets):
        self.article = article
        self.targets = targets
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.category_dic = utils.category_dict()

    def __len__(self):
        return len(self.article)

    def __getitem__(self, item):
        article = utils.normalize(self.article[item])

        inputs = self.tokenizer.encode_plus(article, add_special_tokens=True)

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        padding_len = self.max_len - len(ids)
        if padding_len >= 0:
            ids = ids + [0] * padding_len
            mask = mask + [0] * padding_len
            token_type_ids = token_type_ids + [0] * padding_len
        else:
            half_len = int(self.max_len / 2)
            ids = ids[:half_len] + ids[-half_len:]
            mask = mask[:half_len] + mask[-half_len:]
            token_type_ids = token_type_ids[:half_len] + token_type_ids[-half_len:]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(
                self.category_dic.get(self.targets[item]), dtype=torch.long
            ),
            "orig_targets": self.targets[item],
        }


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop=True)
    dset = LivedoorDataset(article=df.article.values, targets=df.category.values)
    print(dset[0])
