# -- coding:UTF-8
import numpy as np
import librosa
import librosa.display
import torch
import random
import math
import torchaudio
import torchvision
import torchvision.transforms as transforms
from torch.nn import BCEWithLogitsLoss, MSELoss
from torchvision import datasets
from torchvision.datasets import MNIST, CIFAR10
from data_audio import get_train_dl, get_ds, sw_to_stft
from torch.utils.data import Dataset, DataLoader, Subset, DataLoader
import argparse
import options
import matplotlib.pyplot as plt
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Image denoising')).parse_args()
import utils
import os
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from timm.utils import NativeScaler
import time
from losses import CharbonnierLoss
from torchsummary import summary
from glob import glob
from skimage import io, transform
from PIL import Image
from audio_test import SpeechDataTrain, SpeechDataTest, reconstruct_audio, show_watermark, SpeechDataAudio
import soundfile as sf
from pypesq import pesq
import scipy
import scipy.io.wavfile as wavfile
# imports
import matplotlib.pyplot as plt
import numpy as np
import wave, sys

class MultiplyFactor:
    def __init__(self, factor=0.1):
        self.factor = factor

    def __call__(self, image):
        # Assuming image is a PyTorch tensor of shape (3, height, width) with dtype float32
        return image * self.factor

class NormalizeBatch:
    """
    Normalize a batch of tensors to the range [0, 1].

    :param batch: Batch of tensors (B x C x H x W)
    :return: Normalized batch of tensors (B x C x H x W)
    """
    def __init__(self, min_range=0, max_range=1):
        self.min_range = min_range
        self.max_range = max_range

    def __call__(self, image):
        """
        Normalize an image tensor to the given range.

        :param image: Image tensor (C x H x W)
        :param min_range: Minimum value of the normalization range
        :param max_range: Maximum value of the normalization range
        :return: Normalized image tensor (C x H x W)
        """
        c, h, w = image.shape
        reshaped_image = image.view(c, -1)
        min_values = reshaped_image.min(dim=-1, keepdim=True)[0]
        max_values = reshaped_image.max(dim=-1, keepdim=True)[0]
        min_values = min_values.view(c, 1, 1)
        max_values = max_values.view(c, 1, 1)

        normalized_image = (image - min_values) / (max_values - min_values)
        rescaled_image = normalized_image * (self.max_range - self.min_range) + self.min_range
        return rescaled_image
    
def SNR_singlech(S, SN):
    S = S-np.mean(S)# 消除直流分量
    S = S/np.max(np.abs(S))#幅值归一化
    mean_S = (np.sum(S))/(len(S))#纯信号的平均值
    PS = np.sum((S-mean_S)*(S-mean_S))
    PN = np.sum((S-SN)*(S-SN))
    snr=10*math.log((PS/PN), 10)
    return(snr)


def specgram(signal, sample_freq, filename):
    Fs, aud = wavfile.read(signal)
    # select left channel only
    # aud = aud[:,0]
    # trim the first 125 seconds
    plt.figure()
    first = aud[:int(sample_freq*3)]
    plt.specgram(first, Fs=Fs)
    plt.savefig(filename)

# shows the sound waves
def soundwave_visualize(path, save_path):
    raw = wave.open(path)
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="int16")
    f_rate = raw.getframerate()
    time = np.linspace(0, len(signal) / f_rate,num = len(signal))
    plt.figure()
    plt.title("Sound Wave")
    plt.xlabel("Time")
    plt.plot(time, signal)
    plt.savefig(save_path)


def stft_visualize(path, save_path):
    audio_samples, sample_rate = librosa.load(path, sr=None)  # Use sr=None to preserve the original sample rate
    stft_matrix = librosa.stft(audio_samples)
    print('stft matrix: ', stft_matrix.shape)
    magnitude_spectrogram = np.abs(stft_matrix)
    db_spectrogram = librosa.amplitude_to_db(magnitude_spectrogram, ref=np.max)
    print('db matrix: ', db_spectrogram.shape)

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(db_spectrogram, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (STFT)')
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))

def cal_snr(audio_ori, audio_recon):
    min_len = min(len(audio_ori), len(audio_recon))
    power_signal = np.sum(np.square(audio_ori[:min_len]))
    power_noise = np.sum(np.square(audio_ori[:min_len] - audio_recon[:min_len]))
    snr_score = 10 * np.log10(power_signal / power_noise)
    return snr_score

def snr_score(audio):
    singleChannel = np.sum(audio, axis=1)
    norm = singleChannel / (max(np.amax(singleChannel), -1 * np.amin(singleChannel)))
    
    return scipy.stats.signaltonoise(norm)

def cal_pesq(audio_ori, audio_recon, sr):
    print(len(audio_ori), len(audio_recon))
    min_len = min(len(audio_ori), len(audio_recon))
    score = pesq(audio_ori[:min_len], audio_recon[:min_len], sr)
    return score

def pesq_test():
    ori_folder = 'results/audio_gen/test_ori'
    gen_folder = 'results/audio_gen/test'
    files = os.listdir(ori_folder)
    scores = []
    for file in files:
        ref, sr = sf.read(os.path.join(ori_folder, file))
        deg, sr = sf.read(os.path.join(gen_folder, file))
        print(len(ref), len(deg))
        min_len = min(len(ref), len(deg))
        score = pesq(ref[:min_len], deg[:min_len], sr)
        # print(score)
        scores.append(score)
    
    # print(np.mean(scores), np.min(scores), np.max(scores))

def test(model, wm_ds, test_dl, data_cat='train', result_path=None, attack=None, save_audio=True, data_mode='stft', audio_scale='0', data_max=None, data_min=None):
    device = torch.device('cuda')
    wm_dl = iter(DataLoader(wm_ds, batch_size=1, shuffle=False))

    mse_loss_total = []
    wm_loss_total = []
    wm_loss_att_total = []
    snr_total = []
    # snr_recon_total = []
    pesq_total = []
    clips_total = []

    result_file = open('{}/sample_result.txt'.format(result_path), 'a')
    result_file.write('\n')
    for i, data in enumerate(tqdm(test_dl), 0):
        # message, _ = wm_ds[i]
        try:
            message, _ = wm_dl.next()
        except StopIteration:
            wm_dl = iter(DataLoader(wm_ds, batch_size=1, shuffle=False))
            message, _ = wm_dl.next()
        
        # try:
        message = message.to(device)
        ori_audio = data[0][0]
        att_audio, recon_audio, watermark, wm_decode, wm_decode_att, mse_loss, wm_loss, wm_loss_att, snr_ori, snr_recon = reconstruct_audio(data, message, model, attack=attack, data_mode=data_mode, audio_scale=audio_scale, data_max=data_max, data_min=data_min)
        print(att_audio.shape, recon_audio.shape, data[0][0].shape)
        # if attack == 'closed_loop':
        #     pesq_score = pesq(data[0][0].detach().cpu().numpy().reshape(-1), recon_audio.detach().cpu().numpy().reshape(-1), data[0][1].item())
        #     snr_score = snr(data[0][0].detach().cpu().numpy().reshape(-1), recon_audio.detach().cpu().numpy().reshape(-1))
        # else:
        pesq_score = cal_pesq(data[0][0].detach().cpu().numpy().reshape(-1), att_audio, data[0][1].item())
        snr_score = cal_snr(data[0][0].detach().cpu().numpy().reshape(-1), att_audio)
        # pesq_score_ori = pesq(data[0][0].detach().cpu().numpy().reshape(-1), data[0][0].detach().cpu().numpy().reshape(-1), data[0][1].item())
        # print('pesq ori: ', pesq_score_ori, 'pesq recon: ', pesq_score)
        # snr_score_ori = snr(data[0][0].detach().cpu().numpy().reshape(1, -1))
        # snr_score_recon = snr_score(recon_audio.detach().cpu().numpy().reshape(1, -1))
        # print(pesq_score, snr_score_ori, snr_score_recon)
        clips_total.append(len(data[1]))
        mse_loss_total.append(mse_loss)
        wm_loss_total.append(wm_loss)
        wm_loss_att_total.append(wm_loss_att)
        snr_total.append(snr_score)
        # snr_recon_total.append(snr_recon)
        pesq_total.append(pesq_score)
        # torchaudio.save('audio_gen/origianl_{}.wav'.format(i), data[0][0].squeeze(0),
        #                 data[0][1].item())
        if not os.path.exists('{}/audio_gen_sample'.format(result_path)):
            os.mkdir('{}/audio_gen_sample'.format(result_path))
        if not os.path.exists('{}/wm_extract_sample_split'.format(result_path)):
            os.mkdir('{}/wm_extract_sample_split'.format(result_path))
        if not os.path.exists('{}/audio_gen_sample/{}'.format(result_path, data_cat)):
            os.mkdir('{}/audio_gen_sample/{}'.format(result_path, data_cat))
        if not os.path.exists('{}/audio_gen_sample/{}/recon'.format(result_path, data_cat)):
            os.mkdir('{}/audio_gen_sample/{}/recon'.format(result_path, data_cat))
        if not os.path.exists('{}/audio_gen_sample/{}/ori'.format(result_path, data_cat)):
            os.mkdir('{}/audio_gen_sample/{}/ori'.format(result_path, data_cat))
        if not os.path.exists('{}/audio_gen_sample/{}/{}'.format(result_path, data_cat, attack)):
            os.mkdir('{}/audio_gen_sample/{}/{}'.format(result_path, data_cat, attack))
        if not os.path.exists('{}/audio_gen_sample/{}_ori'.format(result_path, data_cat)):
            os.mkdir('{}/audio_gen_sample/{}_ori'.format(result_path, data_cat))
        if not os.path.exists('{}/wm_extract_sample_split/{}'.format(result_path, data_cat)):
            os.mkdir('{}/wm_extract_sample_split/{}'.format(result_path, data_cat))
        if not os.path.exists('{}/wm_extract_sample_split/{}/{}'.format(result_path, data_cat, attack)):
            os.mkdir('{}/wm_extract_sample_split/{}/{}'.format(result_path, data_cat,attack))            
        if save_audio:
            torchaudio.save('{}/audio_gen_sample/{}/recon/{}.wav'.format(result_path, data_cat, i), recon_audio,
                            data[0][1].item())
            
            torchaudio.save('{}/audio_gen_sample/{}/{}/{}.wav'.format(result_path, data_cat,attack, i), torch.from_numpy(att_audio),
                            data[0][1].item())

            torchaudio.save('{}/audio_gen_sample/{}/ori/{}.wav'.format(result_path, data_cat, i), data[0][0].squeeze(),
                            data[0][1].item())
            
            time.sleep(3)
            ref = data[0][0].squeeze().detach().cpu().numpy()
            deg = recon_audio.squeeze().detach().cpu().numpy()
            att = att_audio
            sr = data[0][1].item()
            # ref, sr = sf.read('results/audio_gen_sample/{}/ori/{}.wav'.format(data_cat, i))
            # print(ref.shape, deg.shape, att.shape)
            # deg, sr = sf.read('results/audio_gen_sample/{}/recon/{}.wav'.format(data_cat, i))
            # att, sr = sf.read('results/audio_gen_sample/{}/{}/{}.wav'.format(data_cat, attack, i))
            # print(ref.shape)          
            # pesq_score = pesq(ref, deg, sr)
            # pesq_att_score = pesq(ref, att, sr)
            # snr_ori = signaltonoise(ref)
            # snr_recon = signaltonoise(deg)
            # snr_score = SNR_singlech(ref, deg) 
            # snr_att_score = SNR_singlech(ref, att)

            # result_txt.write('attack:{}, mse:{}, mse attack:{}, pesq:{}, pesq attack:{}\n'.format(attack, wm_loss, wm_loss_att,pesq_score, pesq_att_score))



            soundwave_visualize('{}/audio_gen_sample/{}/recon/{}.wav'.format(result_path, data_cat, i), '{}/audio_gen_sample/{}/recon/{}_soundwave.png'.format(result_path, data_cat, i))
            soundwave_visualize('{}/audio_gen_sample/{}/{}/{}.wav'.format(result_path, data_cat,attack, i), '{}/audio_gen_sample/{}/{}/{}_soundwave.png'.format(result_path, data_cat, attack, i))
            soundwave_visualize('{}/audio_gen_sample/{}/ori/{}.wav'.format(result_path, data_cat, i), '{}/audio_gen_sample/{}/ori/{}_soundwave.png'.format(result_path, data_cat, i))
            stft_visualize('{}/audio_gen_sample/{}/recon/{}.wav'.format(result_path, data_cat, i), '{}/audio_gen_sample/{}/recon/{}_stft.png'.format(result_path, data_cat, i))
            stft_visualize('{}/audio_gen_sample/{}/{}/{}.wav'.format(result_path, data_cat,attack, i), '{}/audio_gen_sample/{}/{}/{}_stft.png'.format(result_path, data_cat, attack, i))
            stft_visualize('{}/audio_gen_sample/{}/ori/{}.wav'.format(result_path, data_cat, i), '{}/audio_gen_sample/{}/ori/{}_stft.png'.format(result_path, data_cat, i))
            specgram('{}/audio_gen_sample/{}/recon/{}.wav'.format(result_path, data_cat, i), sr, '{}/audio_gen_sample/{}/recon/{}_specgram.png'.format(result_path, data_cat, i))
            specgram('{}/audio_gen_sample/{}/{}/{}.wav'.format(result_path, data_cat,attack, i), sr, '{}/audio_gen_sample/{}/{}/{}_specgram.png'.format(result_path, data_cat, attack, i))
            specgram('{}/audio_gen_sample/{}/ori/{}.wav'.format(result_path, data_cat, i), sr, '{}/audio_gen_sample/{}/ori/{}_specgram.png'.format(result_path, data_cat, i))



            show_watermark(watermark, wm_decode, wm_decode_att, '{}/wm_extract_sample_split/{}/{}/wm_{}'.format(result_path, data_cat, attack, i))
        # except:
        #     pass
    print('Result on {} set, attack: {}: Total clips: {}, MSE loss {}, WM loss: {}, WM loss after attack: {}'.format(data_cat, attack, np.sum(clips_total), np.mean(mse_loss_total), np.mean(wm_loss_total), np.mean(wm_loss_att_total)))
    print()

    if result_file:
        result_file.write('Result on {} set, attack: {}: Total clips: {}, MSE loss {}, WM loss: {}, WM loss after attack: {}, SNR score: {}, PESQ score: {}\n'.format(data_cat, attack, np.sum(clips_total), np.mean(mse_loss_total), 
                                                                                                                                                                np.mean(wm_loss_total), np.mean(wm_loss_att_total), 
                                                                                                                                                                np.mean(snr_total), np.mean(pesq_total)))
        result_file.flush()


def model_test():
    # result_path = 'results/ablation_awgn'
    # result_path = 'results/Uformer_audio-tedlium-15062023_151114'
    # result_path = 'results/Uformer_audio-audio-librispeech'
    # result_path = 'results/current_best_early_stop'
    result_path = 'results/Uformer_audio-tedlium-10052023_235146'
    data_mode = 'stft'
    # data_mode = 'audio'

    # watermark process
    transform = transforms.Compose([
                    transforms.Pad(2),
                    transforms.ToTensor(),
                    MultiplyFactor(opt.image_scale),
                    NormalizeBatch(min_range=0, max_range=1),])
    # transform = transforms.Compose([
    #                 transforms.Pad(2),
    #                 transforms.RandomRotation(degrees=(0, 180)),
    #                 transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
    #                 transforms.ToTensor()])
    wm_ds = MNIST(root='../datasets', train=True, download=True, transform=transform)

    # transform = transforms.Compose([
    #         # transforms.Pad(2),
    #         transforms.ToTensor()])
    # wm_ds = CIFAR10(root='../datasets', train=True, download=True, transform=transform)

    dataset_training = SpeechDataTrain(audio_scale=opt.audio_scale, dataset=opt.dataset)
    #print(dataset_training.data_min, dataset_training.data_max)
    time.sleep(10)
    if data_mode == 'stft':
        dataset_train = SpeechDataTest(data_cat='train', size=5, dataset=opt.dataset)
        dataset_test = SpeechDataTest(data_cat='test', size=5, dataset=opt.dataset)
        if opt.audio_scale:
            dataset_train.data_min = dataset_training.data_min
            dataset_train.data_max = dataset_training.data_max
            dataset_test.data_min = dataset_training.data_min
            dataset_test.data_max = dataset_training.data_max
        dataset_train.data = dataset_train.prepare_data(opt.audio_scale)
        dataset_test.data = dataset_test.prepare_data(opt.audio_scale)


    elif data_mode == 'audio':
        dataset_train = SpeechDataAudio(size=30, data_cat='train', data_mode='test')
        dataset_test = SpeechDataAudio(size=30, data_cat='test', data_mode='test')


    # load model
    # opt.arch = 'WMNetCNNAudio'


    model = utils.get_arch(opt)
    model.load_state_dict(torch.load('{}/model.pth'.format(result_path)))
    if opt.audio_scale != '0':
        # print(dataset_training.data_max, dataset_training.data_min)
        model.data_max = dataset_training.data_max
        model.data_min = dataset_training.data_min
    model = model.cuda()
    model.eval()
    # data_set = ['audio']
    # data_cats = ['test']
    data_cats = ['train', 'test']
    # attacks = ['amplitude_scaling-0.8','amplitude_scaling-0.9',
    #             'amplitude_scaling-1.1', 'amplitude_scaling-1.2',
    #             'awgn-5', 'awgn-20',]
    # attacks = ['amplitude_scaling-0.8','amplitude_scaling-0.9',
    #             'amplitude_scaling-1.1', 'amplitude_scaling-1.2',
    #             'awgn-5', 'awgn-20', 'aac', 'mp3compress-64k', 'mp3compress-128k', 'resampling', 'pitch_scaling', 'requantization',
    #              'time_scaling-0.8','time_scaling-0.9', 'time_scaling-1.1', 'time_scaling-1.2']
    attacks = ['jittering_2-100', 'jittering_2-10000']
    # attacks = ['time_scaling-0.8','time_scaling-0.9', 'time_scaling-1.1', 'time_scaling-1.2']
    
    # attacks = ['aac', 'mp3compress-64k', 'mp3compress-128k', 'resampling', 'pitch_scaling', 'requantization',
    #              'time_scaling-0.8','time_scaling-0.9', 'time_scaling-1.1', 'time_scaling-1.2']
    for attack in attacks:
        for data_cat in data_cats:
            if data_cat == 'train':
                test_dl = DataLoader(dataset_train, batch_size=1, shuffle=False)
            else:
                test_dl = DataLoader(dataset_test, batch_size=1, shuffle=False)
            if opt.audio_scale:
                test(model, wm_ds, test_dl, data_cat=data_cat, attack=attack, result_path=result_path,
                  data_mode=data_mode, audio_scale=opt.audio_scale, data_max=dataset_training.data_max, data_min=dataset_training.data_min)
            else:
                test(model, wm_ds, test_dl, data_cat=data_cat, attack=attack, result_path=result_path,
                  data_mode=data_mode, audio_scale=opt.audio_scale)
            
if __name__ == "__main__":
    # pesq_test()
    model_test()
    # model = torch.load('WMNetCNN._e2e_noise_gen_stft_tf.pth')
    # print(model)


