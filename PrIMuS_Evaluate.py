import torch
import torch.utils.data as Data
from torchvision import transforms
from torch.nn import CTCLoss
import torch.nn.functional as F

from PrIMuS_Network import PrIMuS_Network

from PrIMuS_Dataset import PrIMuS_Dataset, PrIMuS_collate_fn, WidthPad
from PrIMuS_CTCDecoder import ctc_decode

import argparse
import os
import numpy as np
from tqdm import tqdm

def min_dis(target, source):
    target = [i for i in target]
    source = [i for i in source]
    target.insert(0, "#")
    source.insert(0, "#")
    sol = np.zeros((len(source), len(target)))

    sol[0] = [i for i in range(len(target))]
    sol[:, 0] = [i for i in range(len(source))]

    for c in range(1, len(target)):
        for r in range(1, len(source)):
            if target[c] != source[r]:
                sol[r, c] = min(sol[r - 1, c], sol[r, c - 1]) + 1
            else:
                sol[r, c] = sol[r - 1, c - 1]

    return sol[len(source) - 1, len(target) - 1]

def error_matric(preds, targets):
    distance = min_dis(preds, targets)
    sequence_error, symbol_error = 0, 0

    if distance != 0:
        sequence_error = 1
        symbol_error = distance / len(targets)

    return sequence_error, symbol_error

def main():
    parse = argparse.ArgumentParser(description="PrIMuS predict")
    
    # predict setting
    parse.add_argument("--model-index", type=int, required=True)
    
    # data setting
    parse.add_argument("--dataset-path", type=str, default="./dataset")
    parse.add_argument("--dataset-type", type=str, default="semantic")
    parse.add_argument("--resize-height", type=int, default=128)
    parse.add_argument("--dataset", type=str, required=True, nargs='+')

    # model setting
    parse.add_argument("--batch-size", type=int, default=1)
    parse.add_argument("--leaky-relu", type=float, default=0.2)
    parse.add_argument("--rnn-hidden", type=int, default=256)

    # model save & validate setting
    parse.add_argument("--save-path-root", type=str, default="./model")
    parse.add_argument("--save-path", type=str, required=True, help="save model in the model dir, and gave name to each model")
    
    # CTC Loss setting
    parse.add_argument("--decode-method", type=str, default="beam_search")
    parse.add_argument("--beam-size", type=int, default=10)

    # set args
    args = parse.parse_args()

    # check model type path is exist, exist -> exit, not exist -> create one
    model_path = os.path.join(args.save_path_root, args.save_path)
    if not os.path.isdir(model_path):
        print("Error: The model path have NOT existed!, Please change --save-path to a new one")
        return

    # is cuda
    use_cuda = torch.cuda.is_available()

    # set device
    device = torch.device("cuda" if use_cuda else "cpu")

    # set predict dict
    evaluate_kwargs = {
        'batch_size': args.batch_size,
        'collate_fn': PrIMuS_collate_fn
    }

    # if use cuda?
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 0,
            'pin_memory': True,
        }
        evaluate_kwargs.update(cuda_kwargs)

    # create transrform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        WidthPad(),
        transforms.ToTensor(),
    ])
    
    evaluate_dataset = PrIMuS_Dataset(
        root=args.dataset_path, 
        split="test",
        type=args.dataset_type,
        datasets=args.dataset,
        resize_height=args.resize_height,
        transform=transform
    )

    # create train/evaluate dataloader
    evaluate_loader = Data.DataLoader(evaluate_dataset, **evaluate_kwargs)
    print("====== end loading dataset... ======")

    # number of classfication (add blank + 1)
    num_class = evaluate_dataset.classfication_num()

    # set CTC loss
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    print("====== start loading model... ======")
    # start init net
    model = PrIMuS_Network(
        args.rnn_hidden, 
        args.leaky_relu,
        num_class,
    )

    # load model
    path = os.path.join(model_path, "PrIMuS_Model_{}.pt".format(args.model_index))
    model.load_state_dict(torch.load(path, map_location=device))

    # set cuda or cpu
    model.to(device)
    print("====== end loading model... ======")

    # set model to evaluate
    print("====== start predict... ======")
    model.eval()

    # set index to name dict
    index_to_name = evaluate_dataset.index_to_name()

    # set progress bar
    pbar = tqdm(total=len(evaluate_loader), desc="Predict")

    # start batch predict (size is evaluate_loader (the file number in package))
    tot_count = len(evaluate_loader)
    tot_loss = 0

    # init sequence/symbol error rate
    sequence_error_num, symbol_error_num = 0, 0

    with torch.no_grad():
        print('\n===== result =====')
        for batch_idx, data in enumerate(evaluate_loader):
            # load data
            images, targets, target_lengths = [d.to(device) for d in data]
            
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
            target_lengths = torch.flatten(target_lengths)

            # set loss
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            # calculate total loss
            tot_loss += loss.item()

            # get ctc_decode predict sequence (final answer)
            preds = ctc_decode(log_probs, method=args.decode_method, beam_size=args.beam_size)

            # list to flatten
            preds = np.array(preds).flatten()
            
            # sum error matrics
            sequence_error, symbol_error = error_matric(preds, targets)
            sequence_error_num += sequence_error
            symbol_error_num += symbol_error
            # print(sequence_error_num, symbol_error_num)
            # print(preds, targets)

            pbar.update(1)
        pbar.close()

        # calculate average error matrics
        sequence_error_rate = sequence_error_num / len(evaluate_loader)
        symbol_error_rate = symbol_error_num / len(evaluate_loader)
        print("Sequence Error Rate -> {}".format(sequence_error_rate))
        print("Symbol Error Rate -> {}".format(symbol_error_rate))

        # final print total loss
        avg_loss = tot_loss / tot_count
        print("Average Loss -> {}".format(avg_loss))

if __name__ == '__main__':
    main()