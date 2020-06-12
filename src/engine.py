import torch
import torch.nn as nn
from tqdm import tqdm


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_fn(torch.softmax(outputs, dim=1), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []

    for bi, d in enumerate(data_loader):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        fin_targets.extend(targets.argmax(1).cpu().detach().numpy().tolist())
        fin_outputs.extend(outputs.argmax(1).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
