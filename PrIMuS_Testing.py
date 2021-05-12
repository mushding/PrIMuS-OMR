import torch
import torch.nn.functional as F

from tqdm import tqdm

from PrIMuS_CTCDecoder import ctc_decode

def test(model, device, test_loader, criterion, decode_method, beam_size):
    model.eval()

    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    wrong_cases = []

    max_iter = None
    pbar_total = max_iter if max_iter else len(test_loader)
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if max_iter and i >= max_iter:
                break

            images, targets, target_lengths = [d.to(device) for d in data]

            logits = model(images)
            log_probs = F.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            tot_count += batch_size
            tot_loss += loss.item()

            pbar.update(1)
        pbar.close()

    evaluation = {
        'loss': tot_loss / tot_count,
    }
    print('valid_evaluation: loss={loss}'.format(**evaluation))
    return evaluation
