import torch
from tqdm import tqdm
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference, SimpleInferer


def train(model, train_dataloader, criterion, optimizer):
    torch.cuda.empty_cache()
    model.train()
    epoch_loss = 0
    train_dataloader = tqdm(train_dataloader, ncols=100)

    for step, batch_data in enumerate(train_dataloader):
        step += 1
        optimizer.zero_grad()

        train_inputs = batch_data["petct"].cuda()
        train_labels = batch_data["seg"].cuda()

        train_outputs = model(train_inputs)
        loss = criterion(input=train_outputs, target=train_labels)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        train_dataloader.set_description(desc=f"step: {step}/{len(train_dataloader)}")
        train_dataloader.set_postfix(train_loss=loss.item())
        train_dataloader.update(1)

    epoch_loss /= step
    return epoch_loss


def validate(
    model, val_dataloader, criterion, dice_metric, post_pred, post_label, roi_size
):
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        val_dataloader = tqdm(val_dataloader, ncols=100)
        for step, batch_data in enumerate(val_dataloader):
            step += 1
            val_inputs = batch_data["petct"].cuda()
            val_labels = batch_data["seg"].cuda()
            val_outputs = sliding_window_inference(
                inputs=val_inputs, roi_size=roi_size, sw_batch_size=4, predictor=model
            )

            loss = criterion(input=val_outputs, target=val_labels)
            _val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            _val_labels = [post_label(i) for i in decollate_batch(val_labels)]

            dice_metric(y_pred=_val_outputs, y=_val_labels)

            val_dataloader.set_description(desc=f"step: {step}/{len(val_dataloader)}")
            val_dataloader.set_postfix(
                val_loss=loss.item(), val_dice=dice_metric.aggregate().item()
            )
            val_dataloader.update(1)
            epoch_loss += loss.item()

    epoch_loss /= step
    epoch_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    return epoch_loss, epoch_dice



