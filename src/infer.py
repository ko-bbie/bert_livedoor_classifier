import pandas as pd
import torch
import torch.nn as nn

import config
import dataset
import engine
from model import BertBaseJapanese
from utils import category_dict_from_num


def predict():
    df = pd.read_csv(config.TEST_FILE).fillna("none")

    test_dataset = dataset.LivedoorDataset(df.article.values, df.category.values)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE, num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertBaseJapanese()
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(config.MODEL_PATH))
    outputs, _ = engine.eval_fn(test_data_loader, model=model, device=device)

    return outputs


if __name__ == "__main__":
    outputs = predict()
    dic = category_dict_from_num()
    outputs = [dic.get(output) for output in outputs]
    result = pd.read_csv(config.TEST_FILE)
    result["pred_category"] = outputs
    result.to_csv("result.csv", index=False)
