import torch.nn as nn
import transformers

import config


class BertBaseJapanese(nn.Module):
    def __init__(self):
        super(BertBaseJapanese, self).__init__()
        bertconfig = transformers.BertConfig.from_pretrained("bert-base-uncased")
        bertconfig.vocab_size = 32000
        bertforpretraining = transformers.BertForPreTraining(bertconfig)
        bertforpretraining.load_tf_weights(bertconfig, config.BERT_PATH)
        self.bert = bertforpretraining.bert
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 9)

    def forward(self, ids, mask, token_type_ids):
        _, pooled_output = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids
        )
        bo = self.bert_drop(pooled_output)
        return self.out(bo)


class AlbertBaseJapanese(nn.Module):
    def __init__(self):
        super(AlbertBaseJapanese, self).__init__()
        self.albert = transformers.AlbertModel.from_pretrained(config.ALBERT_PATH)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 9)

    def forward(self, ids, mask, token_type_ids):
        _, pooled_output = self.albert(
            ids, attention_mask=mask, token_type_ids=token_type_ids
        )
        bo = self.drop(pooled_output)
        return self.out(bo)
