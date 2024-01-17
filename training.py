import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def train_epoch(model, optimizer, loader, epoch, device):
    model.train()
    train_loss = 0
    correct = 0
    count = 0
    tloader = tqdm(loader)
    for batch_idx, (images, questions, labels) in enumerate(tloader, start=1):
        images = images.to(device)
        questions = questions.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = forward_pass(model, images, questions)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        train_loss += loss.detach()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().detach()
        count += torch.ones_like(pred).sum().detach()

        if batch_idx % 100 == 0:
            train_accuracy = (100 * correct / count).item()
            mean_train_loss = (train_loss / batch_idx).item()
            tloader.set_description(
                f'Train Epoch: {epoch} Acc: {train_accuracy:.0f}% Loss: {mean_train_loss:.2f}')

    train_loss = (train_loss / batch_idx).item() # loss function already averages within batches
    train_acc = (correct / count).item()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss))

    return train_loss, train_acc


def eval_model(model, loader, device):
    model.eval()

    test_loss = 0
    test_correct = 0
    test_count = 0
    with torch.no_grad():
        tloader = tqdm(loader)
        for batch_idx, (images, questions, labels) in enumerate(tloader, start=1):
            images = images.to(device)
            questions = questions.to(device)
            labels = labels.to(device)
            output = forward_pass(model, images, questions)
            test_loss += F.cross_entropy(output, labels, reduction='mean').detach()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            test_correct += pred.eq(labels.view_as(pred)).sum().detach()
            test_count += torch.ones_like(pred).sum().detach()

    test_loss = test_loss.item() / batch_idx # loss function already averages within batches
    test_correct = test_correct.item()
    test_count = test_count.item()
    test_acc = test_correct / test_count

    print('====> Test set loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, test_correct, test_count,
        100. * test_correct / test_count))

    return test_loss, test_acc


def forward_pass(model, images, questions):
    return model(images, questions)


def complete_train(
        model, optimizer, train_loader, test_loader, systematic_loader, training_epochs, device):

    train_losses = []
    test_losses = []
    systematic_losses = []
    train_accs = []
    test_accs = []
    systematic_accs = []

    for epoch in range(0, training_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, epoch, device)

        test_loss, test_acc = eval_model(model, test_loader, device)
        systematic_loss, systematic_acc = eval_model(model, systematic_loader, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        systematic_losses.append(systematic_loss)
        systematic_accs.append(systematic_acc)

    return (
        train_losses,
        test_losses,
        systematic_losses,
        train_accs,
        test_accs,
        systematic_accs,
    )
