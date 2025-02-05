import torch

def model_train(model, train_loader, epochs, loss_function, optimizer, device, log = False):
    model.train()
    result = []
    for epoch in range(epochs):
        train_loss = 0.0
        for batch_idx, (X, Y) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)
            predictions = model(X)
            loss = loss_function(predictions, Y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
        train_loss /= len(train_loader)
        if log:
            print('Epoch: {}, Train Loss: {:.4f}'.format(epoch, train_loss))
        result.append(train_loss)
    return result

def model_eval(model, test_loader, device):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(device)
            Y = Y.to(device)
            predictions = model(X)
            accuracy += (predictions.argmax(1) == Y).sum().item()
        accuracy /= len(test_loader.dataset)
    return accuracy