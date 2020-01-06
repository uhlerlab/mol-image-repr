import torch

def train_model(trainloader, model, optimizer, loss_fn, acc_fn, target_kw='target', single_batch=False):
    '''Method for training model (updating model params) based on given criterion'''
    
    model.train()

    total_loss = 0
    total_acc = 0
    total_samples = 0

    for sample in trainloader:
        model.zero_grad()
        output = model(sample)

        target = sample[target_kw]
        batch_size = len(target)

        loss = loss_fn(output, target)
        acc = acc_fn(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()*batch_size
        total_acc += acc
        total_samples += batch_size

        if single_batch:
            break

    return {'train_loss': total_loss / total_samples, 'train_acc': total_acc / total_samples}

def evaluate_model(testloader, model, loss_fn, acc_fn, target_kw='target', single_batch=False):
    '''Method for evaluating model based on given criterion'''
    
    model.eval()

    total_loss = 0
    total_acc = 0
    total_samples = 0

    for sample in testloader:
        with torch.no_grad():
            output = model(sample)
            target = sample[target_kw]
            batch_size = len(target)

            loss = loss_fn(output, target)
            acc = acc_fn(output, target)

        total_loss += loss.item()*batch_size
        total_acc += acc
        total_samples += batch_size

        if single_batch:
            break

    return {'test_loss': total_loss / total_samples, 'test_acc': total_acc / total_samples}
