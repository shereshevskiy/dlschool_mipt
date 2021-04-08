import time
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from nltk.translate.bleu_score import corpus_bleu
from tqdm.auto import tqdm


def get_text(x, trg_vocab):
    id_sw = [trg_vocab.stoi['<unk>'], trg_vocab.stoi['<pad>'], trg_vocab.stoi['<sos>']]
    id_eos = trg_vocab.stoi['<eos>']
    ans = []

    for sec in x:
        if id_eos in sec:
            sec = sec[:np.where(sec == id_eos)[0][0]]
        sec = np.delete(sec, [x in id_sw for x in sec])
        sec = [trg_vocab.itos[elem] for elem in sec]
        ans.append(sec)

    return np.array(ans, dtype=object)


def generate_translation(src, trg, model, trg_vocab):
    model.eval()
    output = model(src, trg, 0)
    output = output[1:].argmax(-1)
    original = get_text(trg.T.cpu().numpy(), trg_vocab)
    generated = get_text(output.T.cpu().numpy(), trg_vocab)

    for i in np.random.choice(len(original), 10):
        print('Original: {}'.format(' '.join(original[i])))
        print('Generated: {}'.format(' '.join(generated[i])))
        print()


def bleu_metric(corpus, model, trg_vocab):
    org_text, gen_text = [], []

    model.eval()
    with torch.no_grad():
        for (trg, src), _ in tqdm(corpus, total=len(corpus)):
            output = model(src, trg, 0)
            output = output[1:].argmax(-1)
            org_text.extend(get_text(trg.T.cpu().numpy(), trg_vocab))
            gen_text.extend(get_text(output.T.cpu().numpy(), trg_vocab))

    return corpus_bleu([[text] for text in org_text], gen_text) * 100


def show(log):
    if len(log['time']) > 0:
        n = len(log['time'])
        m, s = log['time'][-1]
        trn_loss, trn_ppl = log['trn_loss'][-1], math.exp(log['trn_loss'][-1])
        val_loss, val_ppl, val_bleu = log['vld_loss'][-1], math.exp(log['vld_loss'][-1]), log['vld_bleu'][-1]
    else:
        n, m, s, trn_loss, trn_ppl, val_loss, val_ppl, val_bleu = 0, 0, 0, 0, 0, 0, 0, 0

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    clear_output(True)
    fig.suptitle("Epoch {:02}: {}m {}s | Train loss: {:.2f}  PPL: {:4.1f} | Val loss: {:.2f}  PPL: {:4.1f}  BLEU: {:.2f}".format(
        n, m, s, trn_loss, trn_ppl, val_loss, val_ppl, val_bleu), fontsize=16)
    ax[0].plot(list(range(1, len(log['trn_tmp']) + 1)), log['trn_tmp'], 'b', label='train loss')
    ax[0].set_xlabel('Batch', fontsize=12)
    ax[0].set_ylabel('Loss', fontsize=12)
    ax[0].set_title('Current epoch loss (train)', fontsize=14)
    if len(log['trn_loss']) > 0:
        ax[1].plot(list(range(1, len(log['trn_loss']) + 1)), log['trn_loss'], 'bo-', label='train')
        train_loss = log['trn_loss'][-1]
    if len(log['vld_loss']) > 0:
        ax[1].plot(list(range(1, len(log['vld_loss']) + 1)), log['vld_loss'], 'ro-', label='valid')
        valid_loss = log['vld_loss'][-1]
    ax[1].set_xlabel('Epoch', fontsize=12)
    ax[1].set_title('Loss history', fontsize=14)
    ax[1].set_ylabel('Loss', fontsize=12)
    if len(log['time']) > 0:
        ax[1].legend()
    ax[2].plot(list(range(1, len(log['vld_bleu']) + 1)), log['vld_bleu'], 'ro-', label='valid')
    ax[2].set_xlabel('Epoch', fontsize=12)
    ax[2].set_ylabel('BLEU', fontsize=12)
    ax[2].set_title('BLEU history (valid)', fontsize=14)
    plt.show()


def train(model, iterator, optimizer, criterion, clip, tf_ratio, log=None):
    model.train()
    output_dim = model.decoder.output_dim
    epoch_loss = 0
    log['trn_tmp'] = []

    for i, ((trg, src), _) in enumerate(iterator):
        optimizer.zero_grad()
        output = model(src, trg, tf_ratio)
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        log['trn_tmp'].append(loss.cpu().item())

        if (i + 1) % 10 == 0:
            show(log)

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, trg_vocab):
    model.eval()
    output_dim = model.decoder.output_dim
    epoch_loss = 0
    org_text, gen_text = [], []

    with torch.no_grad():
        for (trg, src), _ in iterator:
            pred = model(src, trg, 0)
            output = pred[1:].view(-1, output_dim)
            trg_loss = trg[1:].view(-1)
            loss = criterion(output, trg_loss)
            epoch_loss += loss.item()

            output = pred[1:].argmax(-1)
            org_text.extend(get_text(trg.T.cpu().numpy(), trg_vocab))
            gen_text.extend(get_text(output.T.cpu().numpy(), trg_vocab))

    loss = epoch_loss / len(iterator)
    bleu = corpus_bleu([[text] for text in org_text], gen_text) * 100

    return loss, bleu


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def education(model, optimizer, scheduler, criterion, trg_vocab, trn_iter, val_iter, n_epoch=5, clip=5, tf_ratio=0.5):
    best_valid_bleu = 0
    log = {'trn_tmp': [], 'trn_loss': [], 'vld_loss': [], 'vld_bleu': [], 'time': []}

    for epoch in range(n_epoch):
        start_time = time.time()
        train_loss = train(model, trn_iter, optimizer, criterion, clip, tf_ratio, log)
        valid_loss, valid_bleu = evaluate(model, val_iter, criterion, trg_vocab)
        scheduler.step()
        end_time = time.time()
        log['trn_loss'].append(train_loss)
        log['vld_loss'].append(valid_loss)
        log['vld_bleu'].append(valid_bleu)
        log['time'].append(epoch_time(start_time, end_time))
        show(log)

        if valid_bleu > best_valid_bleu:
            best_valid_bleu = valid_bleu
            torch.save(model.state_dict(), 'best-val-model.pt')

    return log
