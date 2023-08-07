import torchaudio
from torch.utils.data import Subset, DataLoader, Dataset
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy
from audio_attack import low_pass_filter, echo_addition, amplitude_scaling, closed_loop, \
    resampling, requantization, awgn, jittering, jittering_2, aac, time_scaling, pitch_scaling, mp3compress
import random
import librosa
from audio_attack import *
import copy
import pywt
# class FinetuneData(Dataset):
#     def __init__(self, wm_ds, speech_ds):
#         self.wm_ds = wm_ds
#         self.speech_ds = speech_ds
    

#     def __len__(self):
#         return len(self.speech_ds)
    
#     def __getitem__(self, index):

def frequency_masking(magnitude_spectrogram, freq_mask_size):
    num_freq_bins = magnitude_spectrogram.shape[0]
    start_freq = np.random.randint(0, num_freq_bins - freq_mask_size)
    masked_spectrogram = torch.clone(magnitude_spectrogram)
    masked_spectrogram[start_freq:start_freq + freq_mask_size, :, :] = 0
    return masked_spectrogram

def normalize_batch(image, min_range=0.01, max_range=0.1):
    c = image.shape[0]
    reshaped_image = image.view(c, -1)
    min_values = reshaped_image.min()
    max_values = reshaped_image.max()
    # print(min_values, max_values)
    # print(min_values.shape, max_values.shape)
    # min_values = min_values.view(c, 1, 1, 1, 1)
    # max_values = max_values.view(c, 1, 1, 1, 1)

    # normalized_image = (image - min_values) / (max_values - min_values)
    # rescaled_image = normalized_image * (max_range - min_range) + min_range
    rescaled_image = image * 0.025
    return rescaled_image, min_values, max_values

class SpeechDataAudio(Dataset):
    def __init__(self, size=300, len_clip=128, data_cat='train', data_mode='train', mode='audio'):
        # self.path = "/home/tongch/data/LibriSpeech"
        self.len_clip = len_clip
        self.size = size
        self.data_cat = data_cat
        self.data_mode = data_mode
        self.audio_mode = mode
        # self.data_raw = torchaudio.datasets.LIBRISPEECH(
        # root=self.path,
        # url="train-clean-100",
        # download=False,
        # )
        self.path = '/home/tongch/data/tedlium'
        self.data_raw = torchaudio.datasets.TEDLIUM(
            root=self.path,
            release='release2',
            subset='train',
            download=False,
            )
        self.data = self.prepare_data()   

    def prepare_data(self):
        device = torch.device('cuda')
        data = []
        if self.data_cat == 'train':
            indice = list(range(self.size))
        else:
            indice = list(range(600, 600+self.size))

        for i, audio_data in enumerate(self.data_raw):
            if i < indice[0]:
                continue
            elif i > indice[-1]:
                break



            if self.data_mode == 'train': 
                if self.audio_mode == 'dwt':
                    soundwave = audio_data[0].numpy()
                    dwt = np.vstack(pywt.dwt(soundwave, 'coif1'))
                    dwt = torch.from_numpy(dwt)
                # print(dwt.shape)
                    len_pad = self.len_clip**2 - dwt.shape[1] % self.len_clip**2
                # soundwave =  F.pad(soundwave, (0, len_pad), mode='constant', value=0)
                    soundwave =  F.pad(dwt, (0, len_pad), mode='constant', value=0)
                    print(soundwave.shape)
                else:
                    soundwave = audio_data[0]
                    len_pad = self.len_clip**2 - soundwave.shape[1] % self.len_clip**2
                    soundwave =  F.pad(soundwave, (0, len_pad), mode='constant', value=0)
                    sample_max = soundwave.max()
                    sample_min = soundwave.min()


                if self.audio_mode == 'dwt':
                    for i in range(soundwave.shape[1] // self.len_clip**2 * 2):
                        start = random.randint(0, soundwave.shape[1] - self.len_clip**2)
                        sample = soundwave[:, start:start+self.len_clip**2]

                        data.append([soundwave[:, start:start+self.len_clip**2]])
                else:
                    for i in range(soundwave.shape[1] // self.len_clip**2):
                        sample = soundwave[:, self.len_clip**2 * i:self.len_clip**2 * (i + 1)]
                        sample = sample.reshape((1, self.len_clip, self.len_clip))
                        data.append(sample)
                    # sample_normed = sample
                        # sample_normed = (sample-sample_min)/(sample_max-sample_min)
                        # print(sample_normed.max(), sample_normed.min())
                        # data.append([sample_normed, sample_max, sample_min])
                    # data.append(sample)
            elif self.data_mode == 'test':
                if self.audio_mode == 'dwt':
                    soundwave = audio_data[0].numpy()
                    dwt = np.vstack(pywt.dwt(soundwave, 'coif1'))
                    dwt = torch.from_numpy(dwt)
                    len_pad = self.len_clip**2 - dwt.shape[1] % self.len_clip**2
                    soundwave =  F.pad(dwt, (0, len_pad), mode='constant', value=0)
                else:
                    soundwave = copy.deepcopy(audio_data[0])
                    # sample_max = soundwave.max()
                    # sample_min = soundwave.min()
                    # print('sample max min: ', sample_max, sample_min)
                    # print(soundwave.shape, self.len_clip**2, soundwave.shape[1] % self.len_clip**2)
                    len_pad = self.len_clip**2 - soundwave.shape[1] % self.len_clip**2
                    print('len_pad: ', len_pad, soundwave.shape[1] % self.len_clip**2)
                    soundwave =  F.pad(soundwave, (0, len_pad), mode='constant', value=0)
                    
                data_single = []
                for i in range(soundwave.shape[1] // self.len_clip**2):
                    sample = soundwave[:, self.len_clip**2 * i:self.len_clip**2 * (i + 1)].to(device)
                    # sample_normed = (sample-sample_min)/(sample_max-sample_min)
                    sample = sample.reshape(1, self.len_clip, self.len_clip)
                    data_single.append(sample)
                data.append([audio_data, data_single, 128**2-len_pad])                  
        # print(data[0].shape)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # sample = sample.permute(0, 3, 1, 2)
        # sample = sample.squeeze(0)
        # print(sample.shape)

        return sample

class SpeechDataTest(Dataset):
    def __init__(self, size=300, len_clip=128, frequency=128, data_cat='test', dataset='librispeech'):
        if dataset == 'librispeech':
            self.path = "/home/tongch/data/LibriSpeech"
            self.data_raw = torchaudio.datasets.LIBRISPEECH(
                root=self.path,
                url="train-clean-100",
                download=False,
                )
        else:
            self.path = '/home/tongch/data/tedlium'
            self.data_raw = torchaudio.datasets.TEDLIUM(
                root=self.path,
                release='release2',
                subset='train',
                download=False,
                )
        self.len_clip = len_clip
        self.frequency = frequency
        self.data_min = None
        self.data_max = None
        self.data_cat = data_cat
        if size == -1:
            self.size = len(self.data_raw)
        else:
            self.size = size
        # self.data = self.prepare_data()

    def prepare_data(self, audio_scale=False):
        device = torch.device('cuda')
        data = []
        if self.data_cat == 'train':
            indice = list(range(self.size))
        else:
            indice = list(range(300, 300+self.size))


        for i, audio_data in enumerate(self.data_raw):
            if i < indice[0]:
                continue
            elif i > indice[-1]:
                break


            soundwave = audio_data[0]
            stft = torch.stft(soundwave, n_fft=self.frequency * 2 - 1,
                              return_complex=False).to(device)

            # padding
            len_pad = self.len_clip - stft.shape[2] % self.len_clip
            stft_2 = F.pad(stft, (0, 0, 0, len_pad), mode='constant', value=0)
            # print(stft_2.shape, stft_2[:, :, 0, :])

            # split
            clip_data = []
            for j in range(stft_2.shape[2] // self.len_clip):
                data_clip = stft_2[:, :, self.len_clip * j:self.len_clip * (j + 1),
                            :]
                # normalize stft feature
                if audio_scale:
                    data_clip = data_clip * 0.025
                    # data_clip = (data_clip - self.data_min) / (self.data_max - self.data_min) 
                    # data_clip = data_clip * (0.1 - 0.01) + 0.01
                    # data_clip = data_clip * (1000 - 500) + 500

                clip_data.append(data_clip.permute(0, 3, 1, 2).squeeze(0))
        # print(data.shape, data.min(), data.max())

        # return data, min_value, max_value
            data.append([audio_data, clip_data, stft.shape[2] % self.len_clip])

        return data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # sample = sample.permute(0, 3, 1, 2)
        # sample = sample.squeeze(0)
        # print(sample.shape)

        return sample

class SpeechDataTrainAug(Dataset):
    def __init__(self, size=500, len_clip=128, frequency=128, transform=None):
        self.path = "/home/tongch/data/LibriSpeech"
        self.len_clip = len_clip
        self.frequency = frequency
        self.size = size
        self.transform=transform
        # self.data = self.prepare_data()
        self.audio = self.prepare_data()

    def prepare_data(self):
        device = torch.device('cuda')
        data_raw = torchaudio.datasets.LIBRISPEECH(
            root=self.path,
            url="train-clean-100",
            download=False,
            )
        data = []
        for i, audio_data in enumerate(data_raw):
            if i == self.size:
                break

            soundwave = audio_data[0]
            data.append(soundwave)
        return data

    def __len__(self):
        return len(self.audio)
        # return 10000

    def __getitem__(self, idx):
        idx = idx % self.size
        if self.transform:
            audio = self.transform(self.audio[idx])
        else:
            audio = self.audio[idx]

        stft = torch.stft(audio, n_fft=self.frequency*2-1,
                    return_complex=False)

        clip_start = random.randint(0, stft.shape[2]-self.len_clip-1)

        stft_clip = stft[:, :, clip_start:clip_start+self.len_clip, :]
        stft_clip = stft_clip.squeeze(0)
        stft_clip = stft_clip.permute(2, 0, 1)

        return stft_clip

class SpeechDataTrain(Dataset):
    def __init__(self, size=300, len_clip=128, frequency=128, transform=None, audio_scale=False, data_type='train', dataset='librispeech'):
        if dataset == 'librispeech':
            self.path = "/home/tongch/data/LibriSpeech"
            self.data_raw = torchaudio.datasets.LIBRISPEECH(
                root=self.path,
                url="train-clean-100",
                download=False,
                )
        else:
            self.path = '/home/tongch/data/tedlium'
            self.data_raw = torchaudio.datasets.TEDLIUM(
                root=self.path,
                release='release2',
                subset=data_type,
                download=False,
                )
        self.transform = transform
        self.len_clip = len_clip
        self.frequency = frequency
        self.size = size
        self.audio_scale = audio_scale
        self.data_type = data_type
        if self.audio_scale:
            self.data, self.data_min, self.data_max = self.prepare_data()
        else:
            self.data = self.prepare_data()


    def prepare_data(self):
        device = torch.device('cuda')
        data = []
        if self.data_type == 'train':
            sample_indice = list(range(self.size))
        else:
            sample_indice = list(range(self.size, self.size*2))
        
        for i in sample_indice:
            audio_data = self.data_raw[i]
        # for i, audio_data in enumerate(self.data_raw):
        #     if i == self.size:
        #         break

            soundwave = audio_data[0]

            # add data augumentation
            # aug = random.randint(0, 8)
            # if aug == 1:
            #     soundwave = torch.from_numpy(pitch_scaling(soundwave.numpy()))
            # elif aug == 3:
            #     soundwave = torch.from_numpy(resampling(soundwave.numpy()))
            # use dwt feature
            # dwt = np.vstack(pywt.dwt(soundwave, 'coif1'))
            # stft = torch.from_numpy(dwt)

            stft = torch.stft(soundwave, n_fft=self.frequency*2-1,
                              return_complex=False).to(device)
            # print(stft.shape)

            # stft_np = scipy.signal.stft(soundwave, nfft=self.frequency * 2 - 1, nperseg=128,return_onesided=True)
            # stft = torch.from_numpy(stft_np[2].view(np.float32)).to(device)

            # padding
            len_pad = self.len_clip - stft.shape[2] % self.len_clip
            stft_2 = F.pad(stft, (0, 0, 0, len_pad), mode='constant', value=0)
            # print(stft_2.shape, stft_2[:, :, 0, :])

            # split
            # clip_data = []
            for j in range(stft_2.shape[2] // self.len_clip):
                data_clip = stft_2[:, :, self.len_clip * j:self.len_clip * (j + 1),
                            :]
                # clip_data.append(data_clip.permute(0, 3, 1, 2).squeeze(0))
                data.append(data_clip)
            
        
        # print(data.shape)
        # print(data.shape)
        if self.audio_scale:
            data = torch.stack(data)
            data, min_value, max_value = normalize_batch(data)
        # print('min: ', min_value, 'max: ', max_value)
        # print(data.shape, data.min(), data.max())

            return data, min_value, max_value
        else:
            return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        transform_flag = random.randint(0, 5)
        if transform_flag == 1:
            sample = frequency_masking(sample[0], freq_mask_size=15)
            sample = sample.unsqueeze(0)

        sample = sample.permute(0, 3, 1, 2)
        sample = sample.squeeze(0)
        # print(sample.shape)

        return sample


def reconstruct_audio(audio_data, watermark, model, n_fft=255, attack=None, data_mode='stft', data_min=None, data_max=None):
    device = torch.device('cuda')
    # print('audio data: ', len(audio_data))
    # with normalization
    clips = audio_data[1]
    # print()
    # sample_max = audio_data[3]
    # print('sample_max: ', sample_max)
    # sample_min = audio_data[4]

    # clips = audio_data[1]


    len_last_clip = audio_data[2]
    preds = []
    wm_losses = []
    wm_losses_att = []
    wms_decode = []
    for i, clip in enumerate(clips):
        # print(clip.max(), clip.min())
        # audio_clip, wm_gen = model.feature_extract(clip, watermark)
        # wm_decode = model.wm_decode(audio_clip)
        
        # audio_clip, _, _, _, wm_decode, _, _ = model(clip, watermark)
        # audio_clip, _, _, wm_decode = model(clip, watermark)
        # losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch([clip, watermark])
        _, (_, audio_clip, wm_decode) = model.validate_on_batch([clip, watermark])


        # rescale to audio value range
        if data_max is not None:
            # audio_clip = (audio_clip - 0.01) / (0.1 - 0.01)
            # audio_clip = audio_clip * (data_max - data_min) + data_min
            audio_clip = audio_clip * 40

        # print(data_max, data_min, audio_clip.max(), audio_clip.min(), clip.max(), clip.min())
            

        # with normalization
        # (sample-sample_min)/(sample_max-sample_min)
        # print(audio_clip.shape, sample_max, sample_min)
        # audio_clip_ori = audio_clip*(sample_max.cuda()-sample_min.cuda()) + sample_min.cuda()
        # print('audio_clip: ', audio_clip.shape)
        # audio_clip, _ = mo
        # print(torch.max(audio_clip), torch.min(audio_clip))

        wms_decode.append(wm_decode.detach().cpu().numpy())
        # audio_clip, _, wm_decode = model(clip, watermark)
        if i != len(clips)-1:
            # preds.append(audio_clip.detach().cpu().numpy())
            if data_mode == 'stft':
                preds.append(audio_clip.detach().cpu().numpy())
            elif data_mode == 'audio':
                preds.append(audio_clip.reshape(audio_clip.shape[0], 1, -1).detach().cpu().numpy())
        else:
            # print('audio_clip: ', audio_clip.shape)
            if data_mode == 'stft':
                preds.append(audio_clip[:,:,:, :len_last_clip].detach().cpu().numpy())
                recon_audio_stft = np.concatenate(preds, axis=3)
                recon_audio_stft = torch.from_numpy(recon_audio_stft).squeeze(0).permute(1, 2, 0)
                recon_audio = torch.istft(recon_audio_stft, n_fft=n_fft,
                              length=audio_data[0][0].shape[-1],
                              return_complex=False)
            elif data_mode == 'audio':
                preds.append(audio_clip.reshape(audio_clip.shape[0], 1, -1).detach().cpu().numpy()[:, :, :len_last_clip])
                recon_audio = np.concatenate(preds, axis=2)
                recon_audio = torch.from_numpy(recon_audio).to(device)
            elif data_mode == 'dwt':
                preds.append(audio_clip[:, :, :len_last_clip].detach().cpu().numpy())
                recon_dwt = np.concatenate(preds, axis=2)
                # print('recon_dwt: ', recon_dwt.shape)
                recon_audio = pywt.idwt(recon_dwt[0, 0, :], recon_dwt[0, 1,:], 'coif1', 'smooth')
                recon_audio = torch.from_numpy(recon_audio).squeeze().to(device)
                # print('recon_audio: ', recon_audio.shape)
        # print('watermark loss: ', torch.nn.MSELoss()(watermark, wm_decode).item())

    # print(audio_data[0][0].shape, recon_audio.shape)
    if data_mode == 'stft':
        mse_loss = torch.nn.MSELoss()(audio_data[0][0].squeeze(), recon_audio).item()
    elif data_mode == 'audio' or data_mode == 'dwt':
        # print(audio_data[0][0].shape, recon_audio.shape)
        mse_loss = torch.nn.MSELoss()(audio_data[0][0].squeeze().to(device), recon_audio.squeeze()).item()

    # print('audio mse loss: ', mse_loss)
    wm_losses.append(torch.nn.MSELoss()(watermark, wm_decode).item())

    ### apply attack ###
    # audio_att = amplitude_scaling(recon_audio.detach().cpu().numpy())
    if data_mode == 'audio':
        recon_audio = recon_audio.squeeze()
    attack_params = attack.split('-')
    if attack_params[0] == 'echo_addition':
        audio_att = echo_addition(recon_audio.detach().cpu().numpy())
    elif attack_params[0] == 'amplitude_scaling':
        audio_att = amplitude_scaling(recon_audio.detach().cpu().numpy(), factor=float(attack_params[1]))
    elif attack_params[0] == 'low_pass':
        audio_att = low_pass_filter(recon_audio.detach().cpu().numpy())
    elif attack_params[0] == 'closed_loop':
        audio_att = closed_loop(recon_audio.detach().cpu().numpy())
    elif attack_params[0] == 'awgn':
        audio_att = awgn(recon_audio.detach().cpu().numpy(), SNRdB=float(attack_params[1]))

    elif attack_params[0] == 'resampling':
        audio_att = resampling(recon_audio.detach().cpu().numpy())
    elif attack_params[0] == 'requantization':
        audio_att = requantization(recon_audio.detach().cpu().numpy())
    elif attack_params[0] == 'aac':
        audio_att = aac(recon_audio.detach().cpu().numpy())
    elif attack_params[0] == 'jittering':
        audio_att = jittering(recon_audio.detach().cpu().numpy())
    elif attack_params[0] == 'jittering_2':
        audio_att = jittering_2(recon_audio.detach().cpu().numpy())
    elif attack_params[0] == 'time_scaling':
        audio_att = time_scaling(recon_audio.detach().cpu().numpy())
    elif attack_params[0] == 'pitch_scaling':
        audio_att = pitch_scaling(recon_audio.detach().cpu().numpy())
    elif attack_params[0] == 'mp3compress':
        audio_att = mp3compress(recon_audio.detach().cpu().numpy())

    # print('len: ', recon_audio.shape, audio_att.shape)
    if data_mode == 'audio':
        # recon_audio = recon_audio.squeeze(0)
        try:
            audio_att_feat = audio_att.squeeze(0)
        except:
            audio_att_feat = audio_att
    # plt.plot(audio_data[0][0].squeeze().detach().cpu().numpy())
    # plt.savefig('ori_audio.png')
    # plt.plot(recon_audio.detach().cpu().numpy())
    # plt.savefig('recon_audio.png')

    # plt.plot(audio_att)
    # plt.savefig('att_audio.png')
    if data_mode == 'stft':
        audio_att_feat = torch.stft(torch.from_numpy(audio_att), n_fft=255,
                                return_complex=False).to(torch.device('cuda'))
        # split stft feature to 128*128 clips
        # padding
        len_pad = 128 - audio_att_feat.shape[2] % 128
        audio_att_feat = F.pad(audio_att_feat, (0, 0, 0, len_pad), mode='constant', value=0)
        audio_att_feat = audio_att_feat.permute(2, 0, 1).unsqueeze(0)
        # split
        wms_att_decode = []
        for j in range(audio_att_feat.shape[3] // 128):
            data_clip = audio_att_feat[:, :, :, 128 * j:128 * (j + 1)]
            data_clip = data_clip.float()

            # scale to 0-1
            if data_max is not None:
                # data_clip = (data_clip - data_min) / (data_max - data_min)
                # data_clip = data_clip * (0.1 - 0.01) + 0.01
                data_clip = data_clip * 0.025


            # print(torch.max(data_clip), torch.min(data_clip))

            # wm_decode_att = model.wm_decode(data_clip)
            wm_decode_att = model.encoder_decoder.decoder(data_clip)
            # print(torch.max(wm_decode_att), torch.min(wm_decode_att))
            wms_att_decode.append(wm_decode_att.detach().cpu().numpy())
            # print('wm loss after attack: ', torch.nn.MSELoss()(watermark, wm_decode_att).item())
            wm_losses_att.append(torch.nn.MSELoss()(watermark, wm_decode_att).item())

    elif data_mode == 'dwt':
        audio_att_dwt = np.vstack(pywt.dwt(audio_att, 'coif1'))
        audio_att_dwt = torch.from_numpy(audio_att_dwt).to(torch.device('cuda'))
        len_pad = 128**2 - audio_att_dwt.shape[1] % 128**2
                # soundwave =  F.pad(soundwave, (0, len_pad), mode='constant', value=0)
        audio_att_dwt =  F.pad(audio_att_dwt, (0, len_pad), mode='constant', value=0)
        wms_att_decode = []
        # print(audio_att_dwt.shape[1], audio_att_dwt.shape[1] // 128**2)
        for j in range(audio_att_dwt.shape[1] // 128**2):
            data_clip = audio_att_dwt[:, 128**2 * j:128**2 * (j + 1)].to(device)
            data_clip = data_clip.float()
            data_clip = data_clip.unsqueeze(0)
            # print('data_clip: ', data_clip.shape)
            wm_decode_att = model.wm_decode(data_clip)
            wms_att_decode.append(wm_decode_att.detach().cpu().numpy())
            # print('wm loss after attack: ', torch.nn.MSELoss()(watermark, wm_decode_att).item())
            wm_losses_att.append(torch.nn.MSELoss()(watermark, wm_decode_att).item())

    elif data_mode == 'audio':
        audio_att_feat = torch.from_numpy(audio_att_feat).cuda()
        len_pad = 128**2 - len(audio_att_feat) % 128**2
        # insert_indice = [random.randint(0, recon_audio.shape[0]) for _ in range(len_pad)]
        # print('3333:', audio_att.shape)
        # insert_vals = audio_att[insert_indice]
        # print(insert_vals[:10])
        # audio_att = np.insert(audio_att, insert_indice, insert_vals)
        # print('4444:', audio_att.shape)

        # audio_att = torch.from_numpy(audio_att.copy()).unsqueeze(0).to(device)
        audio_att_feat = F.pad(audio_att_feat, (0, len_pad), mode='constant', value=0)
        print('audio_att after pad: ', audio_att_feat.shape)
        # sample_max = audio_att.max()
        # sample_min = audio_att.min()
        # audio_att = (audio_att-sample_min)/(sample_max-sample_min)
        audio_att_feat = audio_att_feat.unsqueeze(0)
        if len(audio_att_feat.shape) == 2:
            audio_att_feat = audio_att_feat.unsqueeze(0)
        wms_att_decode = []
        for j in range(audio_att_feat.shape[2] // 128**2):
            data_clip = audio_att_feat[:, :, 128**2*j:128**2*(j+1)]
            data_clip = data_clip.float()
            # print('data clip att: ', data_clip.shape)
            data_clip = data_clip.reshape(1, 1, 128, 128)
            try:
                wm_decode_att = model.wm_decode(data_clip)
            except:
                wm_decode_att = model.encoder_decoder.decoder(data_clip)
            # print(torch.max(wm_decode_att), torch.min(wm_decode_att))
            wms_att_decode.append(wm_decode_att.detach().cpu().numpy())
            # print('wm loss after attack: ', torch.nn.MSELoss()(watermark, wm_decode_att).item())
            wm_losses_att.append(torch.nn.MSELoss()(watermark, wm_decode_att).item())
    


    # print(wm_losses_att)
        # show_watermark(watermark, wm_decode_att)
    # print(1)
    # print('wm loss: ', np.mean(wm_losses), np.mean(wm_losses_att))
    if data_mode == 'audio':
        print('recon audio, audio att: ', recon_audio.shape, audio_att.shape)
        recon_audio = recon_audio.squeeze().detach().cpu()
        audio_att = audio_att.squeeze()
    elif data_mode == 'dwt':
        print(recon_audio.shape, audio_att.shape)
        recon_audio = recon_audio.detach().cpu()
        # audio_att = audio_att.detach().cpu().numpy()   

    audio_snr_ori = signaltonoise(audio_data[0][0].squeeze())
    audio_snr_recon = signaltonoise(recon_audio)             
    return audio_att, recon_audio, watermark.detach().cpu().numpy(), wms_decode, wms_att_decode, mse_loss, \
            np.mean(wm_losses), np.mean(wm_losses_att), audio_snr_ori, audio_snr_recon


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))

def show_watermark(wm, wms_gen, wms_decode, filename=None, title=None):
    # print(wm[0].shape, wms_gen[0].shape, wms_decode[0].shape)
    message_im = np.squeeze(wm)
    print(len(wms_gen), len(wms_decode))
    min_len = min(len(wms_gen), len(wms_decode))
    plt.imshow(message_im)
    plt.tight_layout()
    plt.savefig(filename + '_original.jpg')
    plt.cla()
    for i in range(min_len):
        message_gen = np.squeeze(wms_gen[i])
        message_decode = np.squeeze(wms_decode[i])
        plt.imshow(message_gen)
        plt.tight_layout()
        plt.savefig(filename + '_no_attack' + str(i+1) + '.jpg')
        plt.cla()
        plt.imshow(message_decode)
        plt.tight_layout()
        plt.savefig(filename + '_after_attack' + str(i+1) + '.jpg')
        plt.cla()
    # for i in range(min_len):
    #     message_gen = np.squeeze(wms_gen[i])
    #     message_decode = np.squeeze(wms_decode[i])
    #     # print(np.max(message_decode), np.min(message_decode))
    #     fig = plt.figure()
    #     axes = []
    #     axes.append(fig.add_subplot(1, 3, 1))
    #     plt.imshow(message_im)
    #     axes.append(fig.add_subplot(1, 3, 2))
    #     plt.imshow(message_gen)
    #     axes.append(fig.add_subplot(1, 3, 3))
    #     plt.imshow(message_decode)

    #     for ax in axes:
    #         ax.axis('off')
        
    #     if title:
    #         plt.title(title)
    #     plt.tight_layout()
    #     plt.savefig(filename + '_' + str(i+1) + '.jpg')
    #     plt.cla()
        # plt.show()

if __name__ == '__main__':
    dataset = SpeechDataTest()
    print(dataset[0])
    # librispeech_path = "/home/tongch/Downloads/LibriSpeech"
    # soundwave_cutoff = 247040
    # train_size = 300
    # batch_size = 1
    # train_ds, _ = get_ds(librispeech_path)
    # train_ds = Subset(train_ds, [i for i in range(train_size)])
    # train_dl = DataLoader(train_ds,
    #                       batch_size=batch_size,
    #                       collate_fn=lambda x: sw_to_stft(x, soundwave_cutoff),
    #                       drop_last=True,
    #                       shuffle=True
    #                       )
    #
    #
    # for sw, stft in train_dl:
    #     stft = torch.squeeze(stft, 1)
    #     istft = torch.istft(stft, n_fft=255, return_complex=False)
    #     print(sw.shape, stft.shape, istft.shape)