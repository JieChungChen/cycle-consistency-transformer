import argparse
import math
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib.pyplot as plt
from model_architecture import RUL_Transformer, Cycle_Consistency_Loss
from preprocessing import RUL_Transformer_Dataset

def get_args_parser():
    parser = argparse.ArgumentParser('Discharge Model Feature Selector training', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--detail_step', default=10, type=int)

    # Model parameters
    parser.add_argument('--model_name', default='RUL_Transformer', type=str) 
    parser.add_argument('--finetune', default=False, type=bool)   
    parser.add_argument('--checkpoint', default='Predictor1.h5', type=str)                  

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    return parser


def main(args):
    model = RUL_Transformer(14, 32).cuda()
    trn_set = RUL_Transformer_Dataset()
    trn_loader = DataLoader(trn_set, batch_size=args.batch_size, num_workers=0, drop_last=True, shuffle=True)
    summary(model,input_size=(1, 100, 14))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = Cycle_Consistency_Loss()

    trn_loss_record = []
    for epoch in range(args.epochs):
        model.train()
        step = 0
        n_minibatch = math.ceil(len(trn_set)/args.batch_size)
        for inputs, src_len in trn_loader:
            step += 1
            optimizer.zero_grad()
            outputs = model(inputs.cuda().float())
            loss = criterion(outputs[0, :src_len[0]], outputs[1, :src_len[1]])
            loss.backward()
            optimizer.step()
            if step%args.detail_step==0:
                print('epoch:[%d / %d] batch:[%d / %d] loss: %.3f lr: %.2e' % (epoch+1, args.epochs, step, n_minibatch, loss, optimizer.param_groups[0]["lr"]))

        if (epoch+1)%100==0:
            torch.save(model.state_dict(), args.model_name+'_ep'+str(epoch+1)+'.pth')

    # training finished
    loss_profile(trn_loss_record)


def loss_profile(trn_loss):
    """
    plot loss v.s. epoch curve
    """
    plt.plot(np.arange(len(trn_loss)), trn_loss, c='blue', label='trn_loss', ls='--')
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.legend()
    plt.show()
    plt.close()


if __name__=='__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)