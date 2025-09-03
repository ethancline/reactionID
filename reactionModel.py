import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time

input_size = 22

class inputLineData(Dataset):
    def __init__(self,data_values,line_parameters):
        if torch.is_tensor(data_values):
            self.data_values = data_values.clone().detach()
        else:
            self.data_values = torch.tensor(data_values , dtype=torch.float32)
        if torch.is_tensor(line_parameters):
            self.line_parameters = line_parameters.clone().detach()
        else:
            self.line_parameters = torch.tensor(line_parameters, dtype=torch.float32)
    def __len__(self):
        return len(self.data_values)
    def __getitem__(self,idx):
        return self.data_values[idx], self.line_parameters[idx] 

class reactionLearner(nn.Module):
    def __init__(self): 
        super(reactionLearner, self).__init__()
        self.sequence = nn.Sequential( 
            nn.Conv2d(2, 8, kernel_size=(10,4), padding=(5,2), bias=False), # 0
            nn.BatchNorm2d(8), # 1 
            nn.ReLU(inplace=True), # 2
            nn.Dropout2d(0.075), # 3

            nn.Conv2d(8, 16, kernel_size=(5,4), padding=(2,2), bias=False), # 4
            nn.BatchNorm2d(16), # 5
            nn.ReLU(inplace=True), # 6
            nn.Dropout2d(0.075), # 7

            nn.Conv2d(16, 1, kernel_size=2), # 8
            nn.ReLU(inplace=True), # 9
            nn.Dropout2d(0.075), # 10

            nn.Flatten(), # 11
            nn.Linear(900, 256, bias=False), # 12
            nn.ReLU(inplace=True), # 13
            nn.BatchNorm1d(256), # 14

            nn.Linear(256, input_size) # 15
        )

    def forward(self, x):
        return self.sequence(x)

def save_model(model, optimizer, epoch, loss, filename):
    checkpoint={'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'epoch' : epoch,
                'loss' : loss} 
    torch.save(checkpoint,filename)

def load_model(model, checkpoint_path):
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded!")
    return model

def MaskedBCELoss(truth, predicted):
    mask = (truth != -1.0)
    t, p = truth[mask], predicted[mask]

    loss = nn.BCEWithLogitsLoss()(p,t) # Model outputs logits
    return loss 

def train_model(model, data, truth, num_epochs=2500, device="cpu", learning_rate=1e-2, savePath="model/default.pth", validData=None, validDecay=None):
    input_data = inputLineData(data_values=data, line_parameters=truth)
    train_loader = DataLoader(input_data, batch_size=128, shuffle=True, drop_last=True, pin_memory=False, num_workers=1, prefetch_factor=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-8)

    # if(os.path.exists(save_path)):
    #     print("Loading previous model...")
    #     model = load_model(model=model, checkpoint_path=save_path)
    #     print("Loaded!")

    if validData is not None:
        valid_data = inputLineData(data_values=validData, line_parameters=validDecay)
        valid_loader = DataLoader(valid_data, batch_size=216, shuffle=True, drop_last=True, pin_memory=False, num_workers=1, prefetch_factor=1)

    losses = []
    valid_losses = [] if validData is not None else None
    lr = []
    print(f'Training...')
    start = time.perf_counter()
    model.train()
    for epoch in range(num_epochs):
        avg_loss = 0.0
        total_samples = 0
        for batch_data, batch_values in train_loader:
            batch_data, batch_values = batch_data.to(device), batch_values.to(device)
            optimizer.zero_grad()
            output = model(batch_data)
            loss = MaskedBCELoss(truth=batch_values, predicted=output)
            loss.backward()
            optimizer.step()            

            batch_size = batch_values.size(0)
            avg_loss += loss.item() * batch_size
            total_samples += batch_size
        avg_loss /= total_samples
        lr.append(scheduler.get_last_lr()[0])
        scheduler.step()
        losses.append(avg_loss)

        if validData is not None:
            model.eval()
            with torch.no_grad():
                avg_loss = 0.0
                total_samples = 0
                for d, v in valid_loader:
                    d, v = d.to(device), v.to(device)
                    out1 = model(d)
                    loss = MaskedBCELoss(truth=v, predicted=out1)
                    batch_size = v.size(0)
                    avg_loss += loss.item() * batch_size
                    total_samples += batch_size 
                avg_loss /= total_samples
                valid_losses.append(avg_loss)
            model.train()
        t_elapsed = time.perf_counter() - start 
        avg_time = t_elapsed/(epoch+1)
        eta = avg_time * (num_epochs - epoch - 1)
        print(f'Epoch {epoch+1}/{num_epochs}. ETA: {round(eta/60,4)} minutes.')
        if((epoch+1) % 5 == 0 or epoch == 0):
            print(f'Learning rate: {lr[-1]}. Loss: {round(losses[-1],5)}')
            if validData is not None:
                print(f'Validation loss: {round(valid_losses[-1],5)}')
    end = time.perf_counter()
    print(f'Trained in {round((end-start)/60,3)} minutes!')
    print(f'Saving model to {savePath}...')
    save_model(model=model,optimizer=optimizer,epoch=num_epochs,loss=losses[-1],filename=savePath)
    print(f'Saved!')
    return model, num_epochs, losses, valid_losses, lr