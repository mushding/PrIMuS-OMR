import torch
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, criterion, epoch, log_interval, is_dry_run):
    
    # set model to train
    model.train()

    # set print loss variable
    total_loss = 0
    total_count = 0

    for batch_idx, (data, target, target_lengths) in enumerate(train_loader):
        # load data
        data, target, target_lengths = data.to(device), target.to(device), target_lengths.to(device)
        
        # set gradient to 0
        optimizer.zero_grad()

        # put data in model
        output = model(data)

        # lstm final layer output probabilty
        log_probs = F.log_softmax(output, dim=2)

        batch_size = data.size(0)
        input_lengths = torch.LongTensor([output.size(0)] * batch_size)
        target_lengths = torch.flatten(target_lengths)

        loss = criterion(log_probs, target, input_lengths, target_lengths)

        loss.backward()
        optimizer.step()

        total_count += batch_size

        # print loss
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))
            total_loss += loss.item()

        if is_dry_run:
            break

    return total_loss / total_count


