from tqdm import tqdm
import argparse
import os
from PIL import Image
import numpy as np
from result_to_midi import result_to_midi

import torch
import torch.utils.data as Data
from torchvision import transforms
import torch.nn.functional as F

from PrIMuS_Network import PrIMuS_Network
from PrIMuS_PredictDataset import PrIMuS_PredictDataset, PrIMuS_collate_fn, WidthPad
from PrIMuS_CTCDecoder import ctc_decode

def main():
    parse = argparse.ArgumentParser(description="PrIMuS predict")
    
    # predict setting
    parse.add_argument("--model-index", type=int, required=True)
    
    # data setting
    parse.add_argument("--dataset-path", type=str, default="./dataset_predict")
    parse.add_argument("--dataset-type", type=str, default="semantic")
    parse.add_argument("--resize-height", type=int, default=128)

    # model setting
    parse.add_argument("--batch-size", type=int, default=1)
    parse.add_argument("--leaky-relu", type=float, default=0.2)
    parse.add_argument("--rnn-hidden", type=int, default=512)

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

    # start init net
    print("====== start loading model... ======")
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

    # set progress bar
    pbar = tqdm(total=len(predict_loader), desc="Predict")

    # set index to name dict
    index_to_name = predict_dataset.index_to_name()

    with torch.no_grad():
        print('\n===== result =====')
        for batch_idx, (data, name) in enumerate(predict_loader):
            # load data
            data = data.to(device)
            
            logits = model(data)
            log_probs = F.log_softmax(logits, dim=2)

            preds = ctc_decode(log_probs, method=args.decode_method, beam_size=args.beam_size, label2char=index_to_name)
            
            # list to string & remove file type (ex: .png) / list to flatten
            name = ''.join(name).split('.')[0]
            preds = np.array(preds).flatten()

            print("Predict filename -> {}".format(name))
            print(preds)

            result_to_midi(preds, name)

            pbar.update(1)
        pbar.close()



if __name__ == '__main__':
    main()