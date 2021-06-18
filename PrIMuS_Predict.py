from tqdm import tqdm
import argparse
import os
from PIL import Image
import numpy as np
from result_to_midi import result_to_midi

import torch
import torch.utils.data as Data
from torchvision import transforms
from torch.nn import CTCLoss
import torch.nn.functional as F

from PrIMuS_ResNet import ResNet_CRNN

from PrIMuS_PredictDataset import PrIMuS_PredictDataset, PrIMuS_collate_fn, WidthPad
from PrIMuS_CTCDecoder import ctc_decode

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
    parse.add_argument("--dataset-path", type=str, default="./dataset_predict")
    parse.add_argument("--midi-path", type=str, default="./midi")
    parse.add_argument("--dataset-type", type=str, default="semantic")
    parse.add_argument("--resize-height", type=int, default=128)

    # model setting
    parse.add_argument("--batch-size", type=int, default=1)
    parse.add_argument("--leaky-relu", type=float, default=0.2)
    parse.add_argument("--rnn-hidden", type=int, default=1024)

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
    predict_kwargs = {
        'batch_size': args.batch_size,
        'collate_fn': PrIMuS_collate_fn
    }

    # if use cuda?
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 0,
            'pin_memory': True,
        }
        predict_kwargs.update(cuda_kwargs)

    # create transrform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        WidthPad(),
        transforms.ToTensor(),
    ])

    # create predict dataset
    print("====== start loading dataset... ======")
    predict_dataset = PrIMuS_PredictDataset(
        root=args.dataset_path, 
        split="predict",
        type=args.dataset_type,
        resize_height=args.resize_height,
        transform=transform
    )

    # create predict dataloader
    predict_loader = Data.DataLoader(predict_dataset, **predict_kwargs)
    print("====== end loading dataset... ======")

    # number of classfication (add blank + 1)
    num_class = predict_dataset.classfication_num()

    # set CTC loss
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    # start init net
    print("====== start loading model... ======")
    model = ResNet_CRNN(
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

    # set progress bar
    pbar = tqdm(total=len(predict_loader), desc="Predict")

    # set index to name dict
    index_to_name = predict_dataset.index_to_name()

    # start batch predict (size is predict_loader (the file number in package))
    tot_count = len(predict_loader)
    tot_loss = 0

    # init sequence/symbol error rate
    sequence_error_num, symbol_error_num = 0, 0

    with torch.no_grad():
        print('\n===== result =====')
        for batch_idx, (data, targets, target_lengths, name, xml_path) in enumerate(predict_loader):
            # load data
            data, targets, target_lengths = data.to(device), targets.to(device), target_lengths.to(device)
            
            logits = model(data)
            log_probs = F.log_softmax(logits, dim=2)

            batch_size = data.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
            target_lengths = torch.flatten(target_lengths)

            # set loss
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            # calculate total loss
            tot_loss += loss.item()

            # get ctc_decode predict sequence (final answer)
            preds = ctc_decode(log_probs, method=args.decode_method, beam_size=args.beam_size, label2char=index_to_name)
            
            print("Predict filename -> {}, Loss -> {}".format(name, loss.item()))
            print(preds)

            # xml_path list to string / list to string & remove file type (ex: .png) / list to flatten
            xml_path = xml_path[0]
            name = ''.join(name).split('.')[0]
            preds = np.array(preds).flatten()

            # result_to_midi(preds, name, xml_path, args.midi_path)

            # sum error matrics
            sequence_error, symbol_error = error_matric(preds, targets)
            sequence_error_num += sequence_error
            symbol_error_num += symbol_error
            print(sequence_error_num, symbol_error_num)
            
            pbar.update(1)
        pbar.close()

        # calculate average error matrics
        sequence_error_rate = sequence_error_num / len(predict_loader)
        symbol_error_rate = symbol_error_num / len(predict_loader)
        print("Sequence Error Rate -> {}".format(sequence_error_rate))
        print("Symbol Error Rate -> {}".format(symbol_error_rate))

        # final print total loss
        avg_loss = tot_loss / tot_count
        print("Average Loss -> {}".format(avg_loss))


if __name__ == '__main__':
    main()