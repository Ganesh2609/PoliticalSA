import sys 
sys.path.append('anonymous/path/to/data')

import torch
from torch import nn
import matplotlib.pyplot as plt

from trainer import ModularTrainer
from dataset import get_data_loaders

from model import PSAModel


def main():


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'



    model = PSAModel().to(device)



    train_dir = "anonymous/path/to/data"
    val_dir = "anonymous/path/to/data"
    test_dir = "anonymous/path/to/data"
    train_loader, test_loader = get_data_loaders(train_dir=train_dir, val_dir=val_dir, test_dir=test_dir, batch_size=4, seed=42)


    learning_rate = 1e-6
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=False)


    trainer = ModularTrainer(
        model=model,
        train_loader=train_loader,  
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        log_path = 'anonymous/path/to/data',
        num_epochs = 32,
        checkpoint_path = "anonymous/path/to/data",
        loss_path = 'anonymous/path/to/data',
        verbose=True,
        device=device
    )

    trainer.train()
    #trainer.train(resume_from="anonymous/path/to/data")


if __name__ == '__main__':
    main()
