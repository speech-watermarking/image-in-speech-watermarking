import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader, Subset, DataLoader
from tqdm import tqdm
import os
import imageio
import itertools

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(42)
np.random.seed(420)
torch.random.manual_seed(42)

def get_ds(librispeech_path):
    train = torchaudio.datasets.LIBRISPEECH(
        root=librispeech_path,
        url="train-clean-100",
        download=False,
        )

    test = torchaudio.datasets.LIBRISPEECH(
        root=librispeech_path,
        url="test-clean",
        download=False)

    return train, test


def align_audio_len(batch, align_len): #247040
    waveform_batch = []
    mark_batch = []
    for i in batch:
        soundwave,mark = i
        mark_batch.append(mark.detach().cpu().numpy())
        waveform_batch.append(torch.transpose(soundwave, 0, 1))
    # waveform_batch = torch.tensor(np.array(waveform_batch)).to(device)
    waveform_batch = pad_sequence(waveform_batch, batch_first=True)
    waveform_batch = waveform_batch.permute(0,2,1)
    # cut the audio into the same length
    waveform_batch = waveform_batch[:,:,:align_len]
    # pad the audio
    if waveform_batch.shape[2] < align_len:
        waveform_batch = F.pad(waveform_batch,(0,align_len-waveform_batch.shape[2]), "constant", 0)
    # stft
    stft_batch = torch.stft(torch.squeeze(waveform_batch), n_fft=1023, win_length=1023, hop_length=511, return_complex=False).to(device)
    stft_batch = stft_batch[:,:,:,0]
    # stft_lib = librosa.stft(waveform_batch[0,:,:].detach().cpu().numpy(),n_fft=1023, win_length=102, hop_length=511)
    # print(stft_lib.shape)
    stft_batch = torch.unsqueeze(stft_batch, dim=1)
    mark_batch = torch.as_tensor(mark_batch)
    return stft_batch,mark_batch


def sw_to_stft(sw_batch, align_len): #247040
    waveform_batch = []
    for i in sw_batch:
        sw = i[0]
        waveform_batch.append(torch.transpose(sw, 0, 1))
    # waveform_batch = torch.tensor(np.array(waveform_batch)).to(device)
    waveform_batch = pad_sequence(waveform_batch, batch_first=True)
    waveform_batch = waveform_batch.permute(0,2,1)
    # cut the audio into the same length
    # waveform_batch = waveform_batch[:,:,:align_len]
    # pad the audio

    # if waveform_batch.shape[2] < align_len:
    #     waveform_batch = F.pad(waveform_batch,(0,align_len-waveform_batch.shape[2]), "constant", 0)
    # stft
    # dct_batch = dct.dct_2d(torch.squeeze(waveform_batch, 1))
    # idct = dct.idct_2d(dct_batch)
    print()
    stft_batch = torch.stft(torch.squeeze(waveform_batch, 1), n_fft=255, return_complex=False).to(device)
    # stft_batch = stft_batch[:,:,:,0]
    # stft_lib = librosa.stft(waveform_batch[0,:,:].detach().cpu().numpy(),n_fft=1023, win_length=102, hop_length=511)
    # print(stft_lib.shape)
    stft_batch = torch.unsqueeze(stft_batch, dim=1)
    # print(stft_batch.shape)
    # stft_batch = stft_batch[:,:,512-128:, :]
    # print(stft_batch)
    return waveform_batch, stft_batch


def get_mark_img(directory):
    mark_patterns, digit_labels = [],[]
    # directory = "img16/"
    for file_label in os.listdir(directory):
        label = os.fsdecode(file_label)
        digit_labels.append(label)
        # if filename.endswith(".asm") or filename.endswith(".py"):
        # print(os.path.join(directory, filename))
        for img in os.listdir(directory + label):
            img_path = directory + label + "/" + img
            im = imageio.imread(img_path)
            im = np.expand_dims(im, axis=0)
            mark_patterns.append(im)
            # print(im.shape)
    return mark_patterns, len(mark_patterns), digit_labels


def gen_watermarking(mark_shape, num, val):
    watermarkings = []
    for i in range(num):
        # mark = np.random.randint(2, size=audios[0].shape) # when mfcc
        # mark = np.random.randint(3, size=[1,33,27]) # when soundwave
        mark = np.random.randint(val, size=mark_shape)
        # if val == 2:
        #     mark[mark==0] = -1
        if val == 3:
            mark[mark==2] = -1
        # print(mark)
        watermarkings.append(mark)
    return watermarkings


class AudioMark(Dataset):
    def __init__(self, soundwav, watermark, markorder):
        self.soundwave = soundwav
        self.watermark = watermark
        self.markorder = markorder

    def __len__(self):
        return len(self.markorder)

    def __getitem__(self, idx):
        sw = self.soundwave[idx]
        mark = self.watermark[self.markorder[idx]]
        sw = torch.tensor(sw).to(device)
        mark = torch.tensor(mark).to(device)
        return (sw, mark)

class AudioMarkAll(Dataset):
    def __init__(self, soundwav, watermark):
        self.soundwave = soundwav
        self.watermark = watermark
        self.pairs = list(itertools.product(self.soundwave, self.watermark))

    def __len__(self):
        return len(self.soundwave)*len(self.watermark)

    def __getitem__(self, idx):
        sw, mark = self.pairs[idx]
        sw = torch.tensor(sw).to(device)
        mark = torch.tensor(mark).to(device)
        return (sw, mark)


def get_img_stats(img_list):
    # imgs: list
    # normalize the img in the list
    image_size = img_list[0].shape[1]
    count = len(img_list) * image_size * image_size
    mark_list = [img / 256 for img in img_list]

    psum, p_sumsq = 0,0
    for img in mark_list:
        psum += np.sum(img)
        p_sumsq += np.sum(np.power(img, 2))

    total_mean = psum / count
    total_var = p_sumsq - np.power(psum, 2) / count
    total_std = np.sqrt(total_var / count)
    return total_mean, total_std


def get_train_dl(librispeech_path, mark_path, soundwave_cutoff, train_size, test_size, mark_shape, watermarking_num, val, batch_size):
    wave_len = 0
    wave_len_all = []
    # get train dataset
    # librispeech_path = "/home/tongch/Downloads/LibriSpeech"
    train_ds, test_ds = get_ds(librispeech_path)

    ############### REMOVED #####################
    # train_subset = Subset(train_ds, range(train_size))
    # test_subset = Subset(train_ds, range(train_size, train_size+test_size))
    # train_sw, test_sw = [], []
    # train_size_count, test_size_sount = 0, 0
    # for i in train_subset:
    #     train_sw.append(i[0].detach().cpu().numpy()[:,:soundwave_cutoff])
    #
    # for i in test_subset:
    #     test_sw.append(i[0].detach().cpu().numpy()[:,:soundwave_cutoff])
    ###########################################

    train_sw_all, test_sw_all = [], []
    train_size_count, test_size_sount = 0, 0
    print("Loading the training set ...")
    for i in tqdm(train_ds):
        if i[0].shape[1] > soundwave_cutoff and train_size_count <= train_size:
            train_sw_all.append(i[0].detach().cpu().numpy()[:, :soundwave_cutoff])
            train_size_count += 1
    print("\nThere are", train_size_count, "audios over 15.44s in training set.")

    print("Loading the test set ...")
    for i in tqdm(test_ds):
        if i[0].shape[1] > soundwave_cutoff and test_size_sount <= test_size:
            test_sw_all.append(i[0].detach().cpu().numpy()[:, :soundwave_cutoff])
            test_size_sount += 1
    print("There are", test_size_sount, "audios over 15.44s in test set.")

    train_sw = train_sw_all[:train_size]
    # test_sw = train_sw_all[train_size:train_size+test_size]
    test_sw = test_sw_all[:test_size]

    ######### REMOVED: 0/1 sequence ##########
    # watermarkings = gen_watermarking(mark_shape, watermarking_num, val)
    ########################################

    mark_patterns, mark_len, digi_labels = get_mark_img(mark_path)
    # mark = torch.tensor(watermarkings, dtype=torch.float32).to(device) # 8 diff watermarkings
    #TODO: change the number of used watermarking
    # normalize
    mark_patterns = [img / 256 for img in mark_patterns]
    # total_mean, total_std = get_img_stats(mark_patterns)
    # mark_patterns = [(img - total_mean) / total_std for img in mark_patterns]

    ## prepare the mark order
    # watermarking_num = len(digi_labels)*10
    # mark_order_train = np.random.randint(watermarking_num, size=train_size)
    # mark_order_test = np.random.randint(watermarking_num, size=test_size)
    #
    # audiomark_train_ds = AudioMark(train_sw, mark_patterns, mark_order_train)
    # audiomark_train_dl = DataLoader(audiomark_train_ds,
    #                           batch_size=batch_size,
    #                           collate_fn=lambda x: align_audio_len(x, soundwave_cutoff),
    #                           drop_last=True,
    #                           shuffle=True)
    #
    # audiomark_test_ds = AudioMark(test_sw, mark_patterns, mark_order_test)
    # audiomark_test_dl = DataLoader(audiomark_test_ds,
    #                           batch_size=batch_size,
    #                           collate_fn=lambda x: align_audio_len(x, soundwave_cutoff),
    #                           drop_last=True,
    #                           shuffle=True)

    audiomark_train_ds = AudioMarkAll(train_sw, mark_patterns)
    print("len of training data:", len(audiomark_train_ds))
    audiomark_train_dl = DataLoader(audiomark_train_ds,
                                    batch_size=batch_size,
                                    collate_fn=lambda x: align_audio_len(x,soundwave_cutoff),
                                    drop_last=True,
                                    shuffle=True)

    audiomark_test_ds = AudioMarkAll(test_sw, mark_patterns)
    audiomark_test_dl = DataLoader(audiomark_test_ds,
                                   batch_size=batch_size,
                                   collate_fn=lambda x: align_audio_len(x,
                                                                        soundwave_cutoff),
                                   drop_last=True,
                                   shuffle=True)


    return audiomark_train_dl, audiomark_test_dl

