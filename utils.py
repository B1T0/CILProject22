from tqdm import tqdm


def train_model(model, scheduler, optimizer, log_dir, dataloader, val_dataloader=None, split=None,
                save_period=10, patience = 20, epochs = 50):

    best_val_loss = None
    early_stopping = 0
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        train_loss = 0
        train_eval = 0
        model.train()
        for batch in tqdm(dataloader):
            loss, eval = model.training_step(batch, 0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
            train_eval += eval
        print(f'Train Loss: {train_loss / len(dataloader)}')
        logging.info(f'Train Loss: {train_loss / len(dataloader)}')
        print(f'Train Eval: {train_eval/len(dataloader)}')
        print(f'Train Eval:{train_eval/len(dataloader)}')
        model.eval()

        if val_dataloader is not None:
            val_eval = 0
            print(f'Validating')
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in tqdm(val_dataloader):
                    loss, eval = model.validation_step(batch, 0)
                    val_loss += loss
                    val_eval += eval
                print(f'Val Loss: {val_loss / len(val_dataloader)}')
                print(f'Val Eval: {val_eval /len(val_dataloader)}')

                if best_val_loss is None:
                    best_val_loss = val_loss
                elif val_loss < best_val_loss:
                    print(f'New best model in epoch {epoch} {best_val_loss}')
                    early_stopping = 0
                    best_val_loss = val_loss
                    logging.info(f'New best model in epoch {epoch} {best_val_loss}')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'loss': train_loss
                    }, log_dir + f'/model_best_{split}.pth')
        scheduler.step()
        if epoch % save_period == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': train_loss
            }, log_dir + f'/model_{epoch}_{split}.pth')
        if val_dataloader is not None:
            early_stopping += 1
            if early_stopping > patience:
                break

    if best_val_loss is not None:
        print(f"Best Val Loss of split {split} {best_val_loss / len(val_dataloader)}")
        logging.info(f"Best Val Loss of split {split} {best_val_loss / len(val_dataloader)}")
        return best_val_loss / len(val_dataloader)
    else:
        return None