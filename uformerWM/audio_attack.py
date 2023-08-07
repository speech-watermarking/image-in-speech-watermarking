import numpy as np
import pydub as pydub
# from numpy import random, zeros, shape, mean
import soundfile as sf
# from pesq import pesq
import scipy
from numpy.lib import math
from scipy.fftpack import dct, idct
from scipy import signal
import librosa
import soundfile as sf
from pydub import AudioSegment
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import resample
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal
import random
import time 
def low_pass_filter(S_watermarked, Fs=16000,low_pass_parameter=8000):
    # low_pass_parameter = input("Please input low_pass_parameter (default=8000):")
    # if low_pass_parameter == 'default':
    #     low_pass_parameter = 8000
    # else:
    #     low_pass_parameter = int(low_pass_parameter)
    wn = 2 * low_pass_parameter / (Fs * 2)
    b, a = signal.butter(8, wn, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    S_extract = signal.filtfilt(b, a, S_watermarked)  # data为要过滤的信号
    return S_extract


def echo_addition(S_watermarked, Fs=16000, td=0.5, AA=0.2):
    # Type 4 - Echo addition #
    # td = input("Please input time delay (default=0.1):")
    # if td == 'default':
    #     td = 0.1
    # else:
    #     td = float(td)
    # AA = input("Please input amplitude (default=0.1):")
    # if AA == 'default':
    #     AA = 0.1
    # else:
    #     AA = float(AA)
    td = float(td)
    AA= float(AA)
    echo_before = np.zeros([int(td * Fs), 1])
    echo_behind = AA * S_watermarked[0: int(len(S_watermarked) - td * Fs)]
    echo = np.append(echo_before, echo_behind)

    S_extract = S_watermarked + echo
    return S_extract


def amplitude_scaling(S_watermarked, factor=0.8):
    factor = float(factor)
    S_extract = S_watermarked * factor
    return S_extract

# def compression(audio_path):
#     song = pydub.AudioSegment.from_wav(audio_path) #wav
#     song.export("S_extract.mp3", format="wav", bitrate="128k") # 64k & 128
#     S_extract, Fs = sf.read("S_extract.mp3")



def closed_loop(S_watermarked):
    # no attacks
    return S_watermarked

def resampling(S_watermarked, fs=16000):
    # TODO: to be tested
    # The watermarked signals are first down-sampled to 22.1 kHz and then up-sampled back to 44.1 kHz
    # soundfile.write("s_watermark_temp.wav", S_watermarked, samplerate=fs/2)
    # y, sr = librosa.load(librosa.ex('trumpet'), sr=fs)
    down_sample = librosa.resample(S_watermarked, orig_sr=fs, target_sr=fs/2)
    s_recover = librosa.resample(down_sample, orig_sr=fs/2, target_sr=fs)
    # print(len(S_watermarked), len(down_sample), len(s_recover))
    # down_sample = resample(S_watermarked, len(S_watermarked) // 2)
    # s_recover = resample(down_sample, len(S_watermarked))
    # s_recover = time_scaling(down_sample, 0.5)

    return s_recover

def requantization(S_watermarked, quantization_bits = 8): # 8 ,16
    # TODO: https://makeabilitylab.github.io/physcomp/signals/QuantizationAndSampling/index.html
    # Re-quantize each sample of the watermarked signals from 16-bits to 8-bits
    # audio_data_16bit = librosa.to_mono(S_watermarked)
    # audio_data_float = audio_data_16bit / (2**16)
    # audio_data_8bit = audio_data_float * (2**8)
    # audio_data_8bit = audio_data_8bit.astype(int)
    # audio_data_8bit = audio_data_8bit.astype(float)
    # y = 100*np.sin(2 * np.pi * freq * x / sampling_rate)
    sf.write('requant.wav', S_watermarked, 16000, subtype='PCM_U8')
    audio_data_8bit, _ = sf.read('requant.wav')
    return audio_data_8bit


def awgn(signal,snr=15):
    """
    Add AWGN noise to input signal.
    
    Parameters:
    signal (np.array): input signal 
    snr (float): desired signal-to-noise ratio (SNR) in dB.
    
    Returns:
    noisy_signal (np.array): signal with added AWGN noise
    """

    # Calculate signal power and convert to dB 
    sig_power = np.mean(signal**2)
    sig_power_db = 10 * np.log10(sig_power)

    # Calculate noise power based on SNR
    noise_power_db = sig_power_db - snr
    noise_power = 10 ** (noise_power_db / 10)

    # Generate white Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    # Add noise to the signal
    noisy_signal = signal + noise

    return noisy_signal

# def awgn(S_watermarked, target_snr_db=15):
#     # TODO: to be tested
#     # The watermarked signals are added with AWGN sequences. The SNRs are set to 20 dB and 15 dB
#     # Set a target SNR
#     # Calculate signal power and convert to dB
#     p_sig = np.mean(np.power(S_watermarked,2))
#     p_awgn = p_sig / np.power(10, (target_snr_db / 10))
#     noise_volts = np.random.normal(size=(len(S_watermarked)))
#     awgn_audio = S_watermarked + np.sqrt(p_awgn) * noise_volts

#     return awgn_audio

def aac(S_watermarked):
    # TODO: to be tested
    # Perform MPEG-4 advanced audio coding based compression on watermarked signals. The compression bit rate is 128 kbps
    try:
        sf.write("acc_temp.wav", S_watermarked, samplerate=16000)
    except:
        S_watermarked = S_watermarked.squeeze()
        sf.write("acc_temp.wav", S_watermarked, samplerate=16000)

    temp_s = AudioSegment.from_file("acc_temp.wav", format="wav")
    temp_s.export("acc_temp.m4a", format="mp4",bitrate="128k")
    audio_mp3 = AudioSegment.from_file("acc_temp.m4a", format="mp4",bitrate="128k")
    audio_np = np.asarray(audio_mp3.get_array_of_samples(),dtype=np.float64)/32768
    # print(S_watermarked[:-5])ff
    # print(audio_np[:-5])
    return audio_np[:len(S_watermarked)]

def jittering(S_watermarked, jit_ratio = 1000):
    # TODO
    indice_to_remove = []
    # for i in range(len(S_watermarked) // jit_ratio):
        # indice_to_remove.append(random.randint(i*jit_ratio, (i+1)*jit_ratio))
    for i in range(jit_ratio):
        indice_to_remove.append(random.randint(0, len(S_watermarked)))
    # indice_to_remove.append(0)

    # print('indice to remove: ', indice_to_remove)
    jit_s = np.delete(S_watermarked, indice_to_remove)
    # S_watermarked[indice_to_remove] = 0
    # jit_s = S_watermarked
    # print(S_watermarked.shape, jit_s.shape)
    # sf.write('att.wav', jit_s, 16000)
    # time.sleep(1)
    # jit_s, _ = sf.read('att.wav')
    return jit_s


def jittering_2(S_watermarked, jit_ratio=1000):
    # TODO
    indice_to_remove = []
    # for i in range(len(S_watermarked) // jit_ratio):
        # indice_to_remove.append(random.randint(i*jit_ratio, (i+1)*jit_ratio))
    for i in range(jit_ratio):
        indice_to_remove.append(random.randint(0, len(S_watermarked)-1))
    # indice_to_remove.append(0)

    # print('indice to remove: ', indice_to_remove)
    # jit_s = np.delete(S_watermarked, indice_to_remove)
    S_watermarked[indice_to_remove] = 0
    jit_s = S_watermarked
    # print(S_watermarked.shape, jit_s.shape)
    # sf.write('att.wav', jit_s, 16000)
    # time.sleep(1)
    # jit_s, _ = sf.read('att.wav')
    return jit_s

def time_scaling(sw, scaling_factor=1):
    # TODO: to be tested
    # http://man.hubwiz.com/docset/LibROSA.docset/Contents/Resources/Documents/generated/librosa.effects.time_stretch.html
    # The watermarked audio signals are time-scaled by a scaling factor of 80%, 90%, 110%, and 120% without shifting the pitch.z
    S_watermarked = librosa.effects.time_stretch(sw, scaling_factor)
    # time_sw = librosa.effects.time_stretch(sw, scaling_factor)
    # sw_padding = np.abs(len(sw) - len(time_sw))
    # if scaling_factor < 1: #speed up: attacked audio short than the original
        # print(len(time_sw_pad))
        # padding_idx = [i for i in range(0, len(time_sw), int(np.ceil(len(time_sw) / sw_padding)))]
        # padding_idx = list(range(0, len(sw), len(sw)//sw_padding))

        # time_sw_pad = np.zeros_like(time_sw)
        # current_frame = 0
        # for i in range(len(time_sw_pad)):
        #     if i in padding_idx:
        #         time_sw_pad[i] = time_sw_pad[i-1]
        #     else:
        #         time_sw_pad[i] = sw[current_frame]
        #         current_frame += 1
        # print('len: ', len(sw), len(time_sw), len(time_sw_pad))
        
        # print('padding_idx: ', padding_idx)
        # print(len(np.unique(padding_idx))==sw_padding)
        # time_sw_pad = np.insert(time_sw, padding_idx, np.nan)
        # print("len time_sw_pad: ", len(time_sw_pad))
        # for i in range(0, len(sw) - 1):
        #     if np.isnan(time_sw_pad[i]) == True:
        #         time_sw_pad[i] = time_sw_pad[i + 1]
    # else: # slow down
    #     # padding_idx= [i for i in range(0, len(sw), int(np.trunc(len(sw)/sw_padding)))]
    #     padding_idx = list(range(0, len(sw), len(sw)//sw_padding))
    #     print(padding_idx[-1], len(sw), len(sw)//sw_padding)
    #     time_sw_pad = np.delete(time_sw, padding_idx)
    #     print(len(np.unique(padding_idx))==sw_padding)
    # return time_sw_pad
    return S_watermarked

def pitch_scaling(S_watermarked, scaling_factor=-6):
    print(S_watermarked.shape)
    # TODO: to be tested
    # notice: replaced with pitch shifting: https://librosa.org/doc/main/generated/librosa.effects.pitch_shift.html
    # The watermarked audio signals are pitch-scaled by a scaling factor of 80%, 90%, 110%, and 120% without modifying the duration of the signals
    S_watermarked = librosa.effects.pitch_shift(S_watermarked, sr=16000, n_steps=scaling_factor)
    return S_watermarked

def mp3compress(S_watermarked,compress_rate="64k"):
    print(S_watermarked.shape)

    # TODO: to be tested
    # compress_rate: 64, 128
    sf.write("mp3_temp.wav", S_watermarked, samplerate=16000)
    # sf_audio = librosa.load("mp3_temp.wav", sr=16000)
    # pydub_audio = AudioSegment.from_file("mp3_temp.wav")
    # pydub_audio_np = np.asarray(pydub_audio.get_array_of_samples(), dtype=np.float64)/32768


    pydun_temp_s = AudioSegment.from_file("mp3_temp.wav", format="wav")
    pydun_temp_s.export("mp4_temp.mp3", format="mp3", bitrate=compress_rate)
    audio_mp3 = AudioSegment.from_file("mp4_temp.mp3", format="mp3")
    audio_mp3_np = np.asarray(audio_mp3.get_array_of_samples(), dtype=np.float64)/32768
    # audio_mp3_np = sf.read("mp4_temp.mp3")
    return audio_mp3_np

def specgram(signal, sample_freq, filename):
    Fs, aud = wavfile.read(signal)
    # select left channel only
    # aud = aud[:,0]
    # trim the first 125 seconds
    plt.figure()
    first = aud[:int(sample_freq*3)]
    plt.specgram(first, Fs=Fs)
    plt.savefig(filename)

if __name__ == "__main__":
    
    # sw, sr = librosa.load("results/audio_gen_sample/train/ori/0.wav", sr=16000)
    # sf.write('ori.wav', sw, 16000)
    # sw_att = resampling(sw)
    # sf.write('att.wav', sw_att, 8000)
    # specgram('att.wav', 000, 'att.png')
    sw, sr = librosa.load("results/audio_gen_sample/train/recon/0.wav", sr=16000)
    # print(sf.info('requant.wav'))
    sf.write('recon.wav', sw, 16000)
    sw_att = jittering(sw)
    sf.write('att_recon.wav', sw_att, 16000)
    specgram('att_recon.wav', 16000, 'att_recon.png')
    specgram('recon.wav', 16000, 'recon.png')

    # print(np.mean(sw))
    # low_pass_sw = low_pass_filter(sw) #0.00184
    # echo_add_sw = echo_addition(sw) # 0.00280
    # req_sw = requantization(sw) #0.0152
    # awgn_sw = awgn(sw) #0.00292
    # aac_sw = aac(sw) #0.00037
    # time_sw = time_scaling(sw) # length not aligned
    # amp_sw = amplitude_scaling(sw, factor=1.1) # 1.2: 0.0030,
    # pitch_sw = pitch_scaling(sw, scaling_factor=12) #0.0195
    # mp3_comp_sw = mp3compress(sw) #
    # print(np.mean(sw))
    # print(np.mean(np.abs(sw-mp3_comp_sw)))


    # print(sw.shape, sw_extracted.shape)