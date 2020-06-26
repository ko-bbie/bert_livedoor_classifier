import torch.nn as nn
import transformers

import config


class BertBaseJapanese(nn.Module):
    def __init__(self):
        super(BertBaseJapanese, self).__init__()
        bertconfig = transformers.BertConfig.from_pretrained("bert-base-uncased")
        bertconfig.vocab_size = 32000
        bertmodel = transformers.BertModel(bertconfig)
        self.bert = bertmodel.load_tf_weights(bertconfig, config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 9)

    def forward(self, ids, mask, token_type_ids):
        _, pooled_output = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids
        )
        bo = self.bert_drop(pooled_output)
        return self.out(bo)
