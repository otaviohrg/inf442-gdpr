import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    model.train()

    train_loss, train_acc = 0, 0
    nan_counter = 0
    for batch_num, batch, in enumerate(dataloader):

        inputs = {"x": batch[0],
                  "mask": batch[1].bool()}

        y_pred = model(**inputs)[0].unsqueeze(dim=0)



        preds = y_pred[0, (batch[3] != pad_token_label_id)[0], :]
        y = (batch[3][0][(batch[3] != pad_token_label_id)[0]])

        loss = loss_fn(preds, y)

        if(torch.isnan(loss)):
            # nan_counter += 1
            continue

        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(preds, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred_class.view(-1))
        if batch_num % 5 == 0:
            print(train_acc/(batch_num+1), "batch: ", batch_num)
            # print((y_pred_class == y).sum().item()/len(y_pred_class.view(-1)))

    train_loss = train_loss / (len(dataloader) - nan_counter)
    train_acc = train_acc / (len(dataloader) - nan_counter)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch_num, batch in enumerate(dataloader):

            inputs = {"x": batch[0],
                      "mask": batch[1].bool()}

            y_pred = model(**inputs)[0].unsqueeze(dim=0)
            preds = y_pred[0, (batch[3] != pad_token_label_id)[0], :]
            y = (batch[3][0][(batch[3] != pad_token_label_id)[0]])

            loss = loss_fn(preds, y)

            if(torch.isnan(loss)):
                continue

            test_loss += loss.item()

            test_pred_labels = preds.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels.view(-1)))
            print(test_acc)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
