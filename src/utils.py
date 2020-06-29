import numpy as np
import torch
import re
import unicodedata


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in (-np.inf, np.inf, -np.nan, np.nan):
            print(
                "Validation score improved ({} --> {}). Saving model!".format(
                    self.val_score, epoch_score
                )
            )
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


def category_dict():
    dic = {
        "sports-watch": [1, 0, 0, 0, 0, 0, 0, 0, 0],
        "smax": [0, 1, 0, 0, 0, 0, 0, 0, 0],
        "dokujo-tsushin": [0, 0, 1, 0, 0, 0, 0, 0, 0],
        "movie-enter": [0, 0, 0, 1, 0, 0, 0, 0, 0],
        "it-life-hack": [0, 0, 0, 0, 1, 0, 0, 0, 0],
        "kaden-channel": [0, 0, 0, 0, 0, 1, 0, 0, 0],
        "peachy": [0, 0, 0, 0, 0, 0, 1, 0, 0],
        "topic-news": [0, 0, 0, 0, 0, 0, 0, 1, 0],
        "livedoor-homme": [0, 0, 0, 0, 0, 0, 0, 0, 1],
    }
    return dic


def category_dict_from_num():
    dic = {
        0: "sports-watch",
        1: "smax",
        2: "dokujo-tsushin",
        3: "movie-enter",
        4: "it-life-hack",
        5: "kaden-channel",
        6: "peachy",
        7: "topic-news",
        8: "livedoor-homme",
    }
    return dic


def normalize(text):
    text = normalize_unicode(text)
    text = capitalize_symbols(text)
    return text


def normalize_unicode(text, form="NFKC"):
    normalized_text = unicodedata.normalize(form, text)
    normalized_text = re.sub(r"\s", " ", normalized_text)
    return normalized_text


def capitalize_symbols(text):
    # this function is often used for mecab morph analysis
    text = text.replace("・", "")
    table = str.maketrans(dict(zip("!?(),.:;/@%&[]", "！？（）、。：；／＠％＆［］")))
    capitalized_text = text.translate(table)
    return capitalized_text
