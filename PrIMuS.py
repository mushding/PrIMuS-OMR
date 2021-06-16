import torch
import torch.utils.data as Data
import torch.optim as optim
from torch.nn import CTCLoss
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms

from PrIMuS_Network import PrIMuS_Network

from PrIMuS_Dataset import PrIMuS_Dataset, PrIMuS_collate_fn, WidthPad
from PrIMuS_Training import train
from PrIMuS_Testing import test

import os
import argparse
import wandb

def main(): 
    # training variables
    parse = argparse.ArgumentParser(description="PrIMuS example")
    parse.add_argument("--batch-size", type=int, default=16, help="input batch size for training (default: 16)")
    parse.add_argument("--test-batch-size", type=int, default=16, help="input batch size for testing (default: 512)")
    parse.add_argument("--epochs", type=int, default=500, help="number of epochs to train (default: 64000)")
    parse.add_argument("--lr", type=float, default=0.0005, help="learning rate (default: 0.0005)")
    parse.add_argument("--log-interval", type=int, default=10, help="how many batches to wait before logging training status")
    parse.add_argument("--dropout", type=float, default=0.5)
    parse.add_argument("--rnn-hidden", type=int, default=512)
    parse.add_argument("--leaky-relu", type=float, default=0.2)
    parse.add_argument("--optimizer", type=str, default="RMSprop")
    parse.add_argument("--scheduler", type=str, default="StepLR")
    parse.add_argument("--checkpoint-path", type=str, default=None)

    # is variables 
    parse.add_argument("--save-model", action="store_true", default=True, help="For Saving the current Model")
    parse.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parse.add_argument("--is-sweep", dest='sweep', action="store_true")
    parse.add_argument("--no-sweep", dest='sweep', action="store_false")

    # data setting
    parse.add_argument("--dataset-path", type=str, default="./dataset")
    parse.add_argument("--dataset-type", type=str, default="semantic")
    parse.add_argument("--dataset", type=str, required=True, nargs='+')
    parse.add_argument("--resize-height", type=int, default=128)

    # CTC Loss setting
    parse.add_argument("--decode-method", type=str, default="beam_search")
    parse.add_argument("--beam-size", type=int, default=10)

    # model save & validate setting
    parse.add_argument("--save-path-root", type=str, default="./model")
    parse.add_argument("--save-path", type=str, required=True, help="save model in the model dir, and gave name to each model")
    parse.add_argument("--valid-interval", type=int, default=5)
    parse.add_argument("--save-interval", type=int, default=1)

    # wandb setting
    parse.add_argument("--is-wandb", dest='wandb', action="store_true")
    parse.add_argument("--no-wandb", dest='wandb', action="store_false")
    parse.add_argument("--wandb-tag", type=str, required=True, nargs='+')

    parse.set_defaults(wandb=True, sweep=False)

    # set args
    args = parse.parse_args()

    # check model type path is exist, exist -> exit, not exist -> create one
    model_path = os.path.join(args.save_path_root, args.save_path)
    if not args.sweep and not args.checkpoint_path:
        if os.path.isdir(model_path):
            print("Error: The model path have already existed!, Please change --save-path to a new one")
            return
        else:
            os.mkdir(model_path) 

    # set seed
    torch.manual_seed(4000)

    # is cuda
    use_cuda = torch.cuda.is_available()

    # set device
    device = torch.device("cuda" if use_cuda else "cpu")

    # set train/test dict
    train_kwargs = {
        'batch_size': args.batch_size,
        'collate_fn': PrIMuS_collate_fn
    }
    test_kwargs = {
        'batch_size': args.test_batch_size,
        'collate_fn': PrIMuS_collate_fn
    }

    # if use cuda?
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 0,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # create transrform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        WidthPad(),
        transforms.ToTensor(),
    ])

    # create train/test dataset
    print("====== start loading dataset... ======")
    train_dataset = PrIMuS_Dataset(
        root=args.dataset_path, 
        split="train", 
        type=args.dataset_type,
        datasets=args.dataset,
        resize_height=args.resize_height,
        transform=transform
    )
    test_dataset = PrIMuS_Dataset(
        root=args.dataset_path, 
        split="test",
        type=args.dataset_type,
        datasets=args.dataset,
        resize_height=args.resize_height,
        transform=transform
    )

    # create train/test dataloader
    train_loader = Data.DataLoader(test_dataset, **train_kwargs)
    test_loader = Data.DataLoader(test_dataset, **test_kwargs)
    print("====== end loading dataset... ======")

    # number of classfication (add blank + 1)
    num_class = train_dataset.classfication_num()

    print("====== start loading model... ======")
    # start init net
    model = PrIMuS_Network(
        args.rnn_hidden, 
        args.leaky_relu,
        num_class,
    )
    print("====== end loading model... ======")

    # check if checkpoint
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    
    # set cuda or cpu
    model.to(device)

    # set optimizer
    optimizer = 0
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    # set lr decrease scheduler
    scheduler = 0
    if args.scheduler == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    elif args.scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif args.scheduler == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # set CTC loss
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    print("====== start training... ======")

    # loss variable
    prev_loss = 10000
    train_loss = 0
    test_loss = 0

    # init wandb
    if args.wandb:
        config = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            learning_rate=args.lr,
            dropout=args.dropout,
            rnn_hidden=args.rnn_hidden,
            optimizer=args.optimizer,
            leaky_relu=args.leaky_relu,
            dataset="PrIMuS",
            architecture="CRNN"
        )
        wandb.init(name=args.save_path, project="PrIMuS", config=config, tags=args.wandb_tag)

    # start training for epoch
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch, args.log_interval, args.dry_run)
        # scheduler.step()

        # every valid_interval (5) test
        evaluation = test(model, device, test_loader, criterion, args.decode_method, args.beam_size)
        test_loss = evaluation['loss']

        if args.wandb:
            wandb.log({"Testing loss": test_loss, "Epoch": epoch, "Training loss": train_loss})
        
        # every save_interval (10) save model & loss smaller than previous train_loss
        if train_loss < prev_loss:
            if args.sweep:
                torch.save(model.state_dict(), os.path.join(model_path, "PrIMuS_Model_{}_{}_{}_{}_{}_{}_{}.pt".format(args.epochs, args.batch_size, args.lr, args.dropout, args.rnn_hidden, args.leaky_relu, args.optimizer)))
            else:
                torch.save(model.state_dict(), os.path.join(model_path, "PrIMuS_Model_{}.pt".format(epoch)))
            if args.wandb:
                wandb.save("PrIMuS_Model_{}_{}_{}_{}_{}_{}_{}.pt".format(args.epochs, args.batch_size, args.lr, args.dropout, args.rnn_hidden, args.leaky_relu, args.optimizer))
            prev_loss = train_loss

    # save the last epoch
    if args.sweep:
        torch.save(model.state_dict(), os.path.join(model_path, "PrIMuS_Model_{}_{}_{}_{}_{}_{}_{}.pt".format(args.epochs, args.batch_size, args.lr, args.dropout, args.rnn_hidden, args.leaky_relu, args.optimizer)))
    else:
        torch.save(model.state_dict(), os.path.join(model_path, "PrIMuS_Model_final.pt"))

if __name__ == "__main__":
    main()
