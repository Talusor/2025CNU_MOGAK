import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import network
import train_and_test as tat

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == 'cuda':
    print(f"Using GPU ({torch.cuda.get_device_name(device)})")

# 랜덤으로 학습데이터중 9개 표시

#figure = plt.figure(figsize=(6, 6))
#for i in range(1, 10):
#    figure.add_subplot(3, 3, i)
#    idx = np.random.randint(len(training_data))
#    plt.title(f"{training_data[idx][1]}")
#    plt.axis('off')
#    plt.imshow(training_data[idx][0].squeeze(), cmap='gray')
#plt.show()

batch_size = 1024


train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 하이퍼파라미터 튜닝은 해당 lr, decay에서 빠르게 loss가 감소하는 모델을 찾는 식으로 진행
# 혹은 tune_loader대신 해당 세트를 분리하여 Validation Set으로 구성하여 Acc측정 하는 방법도 가능
tune_loader = DataLoader(torch.utils.data.Subset(training_data, list(range(1024 * 8))), batch_size=batch_size, shuffle=True)

# 랜덤으로 lr, decay 설정
lrs = [10 ** random.uniform(-4, -1) * random.randint(1, 9) for _ in range(15)]
decays = [10 ** random.uniform(-2, -5) for _ in range(10)]

best_model = None
best_loss = float('inf')
best_params = None
idx = 1
for lr in lrs:
    for decay in decays:
        #model = network.NeuralNetwork().to(device)
        model = network.CNNNetwork().to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
        result = tat.model_train(model=model, train_loader=tune_loader,loss_function=loss_function, optimizer=optimizer, epochs=1, device=device)
        print(f"#{idx} lr: {lr:.5f}, decay: {decay:.5f}, loss: {result[0]:.3f}")
        idx += 1
        if best_loss > result[0]:
            best_loss = result[0]
            best_model = model
            best_params = {"lr": lr, "decay": decay}
print("Done.")
print(f"Best loss: {best_loss:.3f} on lr: {best_params['lr']:.5f} decay: {best_params['decay']:.5f}")

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params["lr"], weight_decay=best_params["decay"])
tat.model_train(model=best_model, train_loader=train_loader,loss_function=loss_function, optimizer=optimizer, epochs=10, device=device, log=True)

accuracy = tat.model_eval(best_model, test_loader, device=device)
print(f"Test Accuracy {accuracy:.3f}")