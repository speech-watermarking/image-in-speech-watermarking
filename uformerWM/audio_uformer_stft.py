import numpy as np
import torch
import random
import json
import torchaudio
import torchvision
import torchvision.transforms as transforms
from torch.nn import BCEWithLogitsLoss, MSELoss
from torchvision import datasets
from torchvision.datasets import MNIST
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
from audio_test import SpeechDataTrain, SpeechDataTrainAug, SpeechDataAudio, SpeechDataTest, reconstruct_audio, show_watermark
from wm_network import spectral_loss
import inspect
import shutil
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)
librispeech_path = "/home/tongch/data/LibriSpeech"
# librispeech_path = "/home/tongch/data/LibriSpeech"
mark_path = "img16/"
soundwave_cutoff = 247040
train_size = 300
test_size = 100
mark_shape = [1, 16, 16]
watermarking_num = 4
val = 2
batch_size = 24
epoch = 300
model_save_dir = "saved_ckpt_01/"
model_save_name = "model"
message_length = 64

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def noise_loss(noise):
        # Step 1: Calculate the mean values of two channels
        mean_values = torch.mean(noise, dim=(2,3))  # resulting shape: (n, 2)

        # Step 2: Calculate the variance between each point and the corresponding mean value
        expanded_mean_values = mean_values.unsqueeze(-1).unsqueeze(-1).expand_as(noise)  # expand mean_values to match the shape of input
        variances = (noise - expanded_mean_values)**2  # variance calculation

        # Step 3: Calculate the mean variance of all points
        mean_variance = torch.mean(variances)

        return mean_variance


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

# ######### Set Seeds ###########
# random.seed(1234)
# np.random.seed(1234)
# torch.manual_seed(1234)
# torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt)
model_restoration.to(device)
# summary(model_restoration, [(1,128,128), (1,32,32)])
# print(model_restoration)

######### Dataset ###########
if 'tedlium' == opt.dataset or 'librispeech' == opt.dataset:
    # train_ds = SpeechDataTrain()
    transform = transforms.Compose([torchaudio.transforms.Resample(new_freq=8000),
                                    # torchaudio.transforms.TimeStretch(),

                                #  torchaudio.transforms.Resample(resample_freq=8000),
                                # torchaudio.transforms.PitchShift(),
                                            ])
    train_ds = SpeechDataTrain(audio_scale=opt.audio_scale, dataset=opt.dataset)
    # train_ds = SpeechDataAudio()

    # train_ds = SpeechDataTrain()

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          drop_last=True)
    # train_ds, _ = get_ds(librispeech_path)
    # train_ds = Subset(train_ds, [i for i in range(train_size)])
    # train_dl = DataLoader(train_ds,
    #                       batch_size=batch_size,
    #                       collate_fn=lambda x: sw_to_stft(x,soundwave_cutoff),
    #                       drop_last=True,
    #                       shuffle=True
    #                       )
# elif 'binary' == opt.dataset:

        

elif 'imagenet' == opt.dataset:
    # get ImageNet
    image_path = "/home/tongch/data/imagenet-mini/train"
    image_files = glob(image_path +  '/*/*.JP*G')
    print("Training sample:", len(image_files))

    class ImageNetMini(Dataset):
        def __init__(self, file_dir, transform=None):
            self.file_dir = file_dir
            self.transform = transform
            self.data = self.get_data()

        def get_data(self):
            data = []
            for i, file in enumerate(self.file_dir):
                if len(data) == 12000:
                    break
                try:
                    image = Image.open(file)
                    image = image.convert(mode='RGB')
                    width, height = image.size
                    if width >= 128 and height >= 128:
                        data.append(image)
                except:
                    pass

            return data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # if torch.is_tensor(idx):
            #     idx = idx.tolist()

            # image = Image.open(self.file_dir[idx])
            # image = image.convert(mode='RGB')
            sample = self.data[idx]
            # try:
            if self.transform:
                sample = self.transform(sample)
            # except ValueError:
                # sample = Image.new('RGB', (128, 128))
                # sample = self.transform(sample)

            return sample

    transforms_imagenet = transforms.Compose([transforms.RandomCrop((128, 128)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                             ])
    imagenet_mini_ds = ImageNetMini(image_files, transform=transforms_imagenet)
    # imagenet_mini_ds = datasets.ImageFolder(
            # "/media/tongch/C8B41AFCB41AED26/imagenet-mini/train", transforms_imagenet)[:train_size]
    train_dl = DataLoader(imagenet_mini_ds, batch_size=batch_size, drop_last=True)


# watermark process
transform = transforms.Compose([
                transforms.Pad(2),
                # transforms.RandomRotation(degrees=(0, 180)),
                # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
                transforms.ToTensor(),
                MultiplyFactor(opt.image_scale),
                NormalizeBatch(min_range=0, max_range=1),
                ])

wm_ds = MNIST(root='../datasets', train=True, download=True, transform=transform)
wm_dl = iter(DataLoader(wm_ds, batch_size=batch_size, shuffle=True))

class BinaryWM(Dataset):
    def __init__(self, num_wm=12000, transform=None):
        self.num_wm = num_wm
        self.transform = transform
        self.wms = self.gen_wm()
    
    def gen_wm(self):
        wms = []
        for i in range(self.num_wm):
            wm = np.random.randint(2, size=32*32).reshape(32, 32, 1)
            wms.append(wm)
        
        return wms

    def __len__(self):
        return len(self.wms)
    
    def __getitem__(self, idx):
        if self.transform:
            x = self.transform(self.wms[idx])
        else:
            x = self.wms[idx]
        return x

# wm_ds = BinaryWM(transform=transforms.ToTensor())
# wm_dl = iter(DataLoader(wm_ds, batch_size=batch_size, shuffle=True, drop_last=True))


######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")

######### DataParallel ###########
# model_restoration = torch.nn.DataParallel(model_restoration)
model_restoration = model_restoration.to(device)
# model_restoration.decoder_wm.to(device)

if opt.audio_scale != '0':
    model_restoration.data_max = train_ds.data_max
    model_restoration.data_min = train_ds.data_min


def fine_tune(model, wm_ds):
    model.load_state_dict(torch.load('WMNetCNN._e2e_no_random.pth'))
    model = model.to(device)
    wm_dl = iter(DataLoader(wm_ds, batch_size=batch_size, shuffle=True))

    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.wm_feature_extractor.parameters():
        param.requires_grad = True
    
    for param in model.wm_decoder.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
    step = 10
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()
    criterion_wm = MSELoss().cuda()
    criterion_audio = MSELoss().cuda()
    loss_scaler = NativeScaler()

    for epoch in range(start_epoch, opt.nepoch + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1

        # for i, (sw, data) in enumerate(tqdm(train_dl), 0):
        for i, batch_data in enumerate(tqdm(train_dl), 0):
            data = batch_data
            # print(data.shape, type(data))
            # zero_grad
            optimizer.zero_grad()
            try:
                message, _ = wm_dl.next()
            except StopIteration:
                wm_dl = iter(DataLoader(wm_ds, batch_size=batch_size, shuffle=True))
                message, _ = wm_dl.next()
            data = data.to(device)
            message = message.to(device)
            # target = data[0].cuda()
            # input_ = data[1].cuda()
            target = data
            input_ = data


            # audio_stft = model.feature_extract(input_, message)
            # audio_stft = audio_stft.permute(0, 2, 3, 1)

            # audio = torch.istft(audio_stft, n_fft=255)

            # stft_2 = torch.stft(audio, n_fft=255)
            # stft_2 = stft_2.permute(0, 3, 1, 2)
            audio, wm_gen = model.feature_extract(input_, message)
            wm = model.wm_decode(audio)
            loss1 = criterion_audio(target, audio)
            loss2 = criterion_wm(wm_gen, message)
            loss3 = criterion_wm(wm, message)
            loss = loss1 + loss2 + loss3
            # loss = criterion_wm(wm, message)
            if i % 20 == 0:
                # print('Epoch {} Step {} loss {}'.format(epoch, i, loss))
                print("Epoch {}, Step {}, Total loss: {}, audio mse loss: {}, Watermark gen loss: {}, watermark decode loss: {}".format(epoch, i, loss, loss1, loss2, loss3))

            loss_scaler(
                    loss, optimizer,parameters=[model.parameters()])

        scheduler.step()

        if epoch % 5 == 0:
            torch.save(model_restoration.state_dict(), 'WMNetCNN_finetune.pth')
            



def train(wm_ds, train_dl):
    wm_dl = iter(DataLoader(wm_ds, batch_size=batch_size, shuffle=True))
    ######### Scheduler ###########
    if opt.warmup:
        print("Using warmup and cosine strategy!")
        warmup_epochs = opt.warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                opt.nepoch - warmup_epochs,
                                                                eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1,
                                           total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)
        scheduler.step()
    else:
        step = 10
        print("Using StepLR,step={}!".format(step))
        scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
        scheduler.step()


    # criterion_wm = BCEWithLogitsLoss().cuda()
    # criterion_wm = MSELoss().cuda()
    criterion_wm = torch.nn.CosineEmbeddingLoss()
    # criterion_wm = MSELoss().cuda()
    # criterion_wm = MSELoss().cuda()
    criterion_audio = MSELoss().cuda()

    # create result folder
    loss_scaler = NativeScaler()
    dt_string = datetime.now().strftime("%d%m%Y_%H%M%S")
    result_path = 'results/{}-{}-{}'.format(opt.arch, opt.dataset, dt_string)
    os.mkdir(result_path)
    os.mkdir(os.path.join(result_path, 'images'))
    os.mkdir(os.path.join(result_path, 'audios'))

    # save exp config
    with open(os.path.join(result_path, 'exp_config.txt'), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)
    
    # save loss
    loss_file = open(os.path.join(result_path, 'exp_loss.txt'), 'w')

    # save model file and training file as a backup
    model_file = inspect.getfile(model_restoration.__class__)
    shutil.copyfile(model_file, os.path.join(result_path, 'model.py'))
    shutil.copyfile('./audio_uformer_stft.py', os.path.join(result_path, 'train.py'))


    loss_best = None
    # add early stop
    loss_best_epoch = 0

    ### watermarker autoencoder warmup ###
    # if opt.warmup:
    #     wm_ds = MNIST(root='../datasets', train=True, download=True, transform=transform)

    #     wm_dl_2 = DataLoader(wm_ds, batch_size=128, shuffle=True)
    #     optimizer_autoencoder =  optim.SGD(model_restoration.parameters(), lr=0.01)
    #     for epoch in range(100):
    #         for i, data in enumerate(tqdm(wm_dl_2), 0):
    #             optimizer_autoencoder.zero_grad()
    #             image = data[0]
    #             image = image.to(device)
        
    #             pred = model_restoration.encoder_wm(image)[1]
        
    #             loss = criterion_wm(image, pred)
        
    #             if i % 10 == 0:
    #                 print(epoch, i, loss)
    #             loss.backward()
    #             optimizer_autoencoder.step()
    #

    for epoch in range(start_epoch, opt.nepoch + 1):
        epoch_loss = 0

        for i, batch_data in enumerate(tqdm(train_dl), 0):
            data = batch_data
            # print(data.shape)
            optimizer.zero_grad()
            try:
                message, _ = wm_dl.next()
                # message = wm_dl.next()

            except StopIteration:
                wm_dl = iter(DataLoader(wm_ds, batch_size=batch_size, shuffle=True, drop_last=True))
                message, _ = wm_dl.next()
                # message = wm_dl.next()

            data = data.to(device)
            # print('stft data type: ', type(data), data.shape)
            # print(data)
            # istft = torch.istft(data.permute(0, 2, 3, 1), n_fft=256, hop_length=128, win_length=256)
            # print(istft.shape)
            # stft_new = torch.stft(istft, n_fft=256, hop_length=128, win_length=256).permute(0, 3, 1, 2)
            # print('difference: ', torch.abs(stft_new - data).max())

            message = message.type(torch.FloatTensor)
            # message = message * image_scale

            message = message.to(device)
            # print(message.shape, data.shape)
            target = data
            input_ = data
            # ablation study 1
            # audio, audio_noise, wm_decode  = model_restoration(input_, message)
            # audio, audio_noise, wm_gen, wm_decode  = model_restoration(input_, message)
            audio, audio_noise, wm_gen, wm_decode  = model_restoration(input_, message)

            # audio, audio_noise, audio_recover, wm_gen, wm_decode, _, _ = model_restoration(input_, message)
            # audio, audio_noise, audio_recover, wm_gen, \
            #  wm_decode, wm_feature, wm_feature_extract = model_restoration(input_, message)

            # audio, audio_noise, wm_decode = model_restoration(input_, message)
            # audio, wm_gen = model_restoration(input_, message)

            # print(audio.size(), wm_gen.size())

            loss1 = criterion_audio(audio, target)
            # loss1 = spectral_loss(target, audio)
            # loss1 = criterion_audio(target, audio)
            # loss1 = 0
            
            # loss4 = max(torch.norm(audio_noise[:, :, :64, :]), 10)
            # loss4 = max(torch.norm(audio_noise), 5) # keep noise with size 30
            noise_norm = torch.norm(audio_noise)/audio_noise.shape[0]
            loss4 = criterion_audio(noise_norm, torch.ones_like(noise_norm))
            # loss5 = criterion_audio(audio_recover, audio)
            loss2 = criterion_audio(wm_gen, message)
            # loss3 = criterion_audio(wm_decode, message) + criterion_audio(wm_decode, wm_gen)
            loss3 = criterion_audio(wm_decode, message)
            # loss5 = noise_loss(audio_noise)

            # loss6 = criterion_audio(wm_feature_extract, wm_feature)
            
            # loss = loss1 + loss2 + loss3 + loss4 + loss5
            # loss = loss1 + loss3 + loss4
            loss = loss1 + loss2 + loss3 + loss4
            # else:
            # loss = loss1 + loss2

            if i % 10 == 0:
                # decoded_rounded = wm_decode.detach().cpu().numpy().round().clip(0, 1)
                # bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (
                #         batch_size * message.shape[1])
                if opt.dataset == 'tedlium' or opt.dataset == 'librispeech':
                    rows = 4
                    columns = 2
                    fig = plt.figure()
                    axes = []
                    for j in range(rows):
                        message_im = message[j].squeeze(0).detach().cpu().numpy()
                        # message_gen = wm_gen[j].squeeze(0).detach().cpu().numpy()
                        message_decode = wm_decode[j].squeeze(0).detach().cpu().numpy()

                        axes.append(fig.add_subplot(rows, columns, j*2+1))
                        plt.imshow(message_im)
                        # axes.append(fig.add_subplot(rows, columns, j*3+2))
                        # plt.imshow(message_gen)
                        axes.append(fig.add_subplot(rows, columns, j*2+2))
                        plt.imshow(message_decode)
                    plt.savefig('{}/images/epoch{}_step{}.png'.format(result_path, epoch, i))                    
                    # plt.savefig('results/1600_mod_imagenet/epoch{}_step{}.png'.format(epoch, i))
                    # plt.savefig('results/{}-{}-{}/epoch{}_step{}.png'.format(opt.arch, opt.dataset, image_scale, epoch, i))
                elif opt.dataset == 'imagenet':
                    rows = 4
                    columns = 4
                    fig = plt.figure()
                    axes = []
                    for j in range(rows):
                        # print(data[j].shape)
                        image_ori = data[j].permute(1, 2, 0).detach().cpu().numpy()
                        axes.append(fig.add_subplot(rows, columns, j*4+1))
                        plt.imshow(image_ori)
                        image_gen = audio[j].permute(1, 2, 0).detach().cpu().numpy()
                        axes.append(fig.add_subplot(rows, columns, j*4+2))
                        plt.imshow(image_gen)
                        message_im = message[j].squeeze(0).detach().cpu().numpy()
                        message_gen = wm_decode[j].squeeze(0).detach().cpu().numpy()

                        axes.append(fig.add_subplot(rows, columns, j*4+3))
                        plt.imshow(message_im)
                        axes.append(fig.add_subplot(rows, columns, j*4+4))
                        plt.imshow(message_gen)

                    plt.savefig('{}/images/epoch{}_step{}.png'.format(result_path, epoch, i))                    

                # print("Epoch {}, Step {}, Total loss: {}, audio mse loss: {}, noise loss: {}, Watermark gen loss: {}, Watermark decode loss: {}".format(epoch, i, loss, loss1, loss4, loss2, loss3))
                # loss_file.write("Epoch {}, Step {}, Total loss: {}, audio mse loss: {}, noise loss: {}, Watermark gen loss: {}, Watermark decode loss: {}\n".format(epoch, i, loss, loss1, loss4, loss2, loss3))

                print("Epoch {}, Step {}, Total loss: {}, audio mse loss: {}, noise loss: {}, Watermark decode loss: {}".format(epoch, i, loss, loss1, loss4, loss3))
                loss_file.write("Epoch {}, Step {}, Total loss: {}, audio mse loss: {}, noise loss: {}, Watermark decode loss: {}\n".format(epoch, i, loss, loss1, loss4, loss3))

            loss_scaler(
                    loss, optimizer,parameters=[model_restoration.parameters()])
            epoch_loss +=loss.item()
        
        loss_file.flush()

            # optimizer.step()
        scheduler.step()
        if loss_best is None or epoch_loss < loss_best:
            torch.save(model_restoration.state_dict(), os.path.join(result_path, 'model.pth'))
            loss_best = epoch_loss
            loss_best_epoch = epoch
        # else:
        #     if epoch - loss_best_epoch == 5:
        #         print('early stop')
        #         break


def test(model, wm_ds):
    result_path = 'results/Uformer_audio-audio-28032023_225128'
    model.load_state_dict(torch.load('{}/model.pth'.format(result_path)))
    model = model.to(device)
    model.eval()
    wm_dl = iter(DataLoader(wm_ds, batch_size=1, shuffle=False))

    dataset_test = SpeechDataTest()
    test_dl = DataLoader(dataset_test, batch_size=1, shuffle=False)


    for i, data in enumerate(tqdm(test_dl), 0):
        # message, _ = wm_ds[i]
        try:
            message, _ = wm_dl.next()
        except StopIteration:
            wm_dl = iter(DataLoader(wm_ds, batch_size=1, shuffle=False))
            message, _ = wm_dl.next()
        message = message.to(device)
        _, recon_audio, watermark, wm_decode_att = reconstruct_audio(data, message, model)
        torchaudio.save('{}/audios/origianl_{}.wav'.format(result_path, i), data[0][0].squeeze(0),
                        data[0][1].item())
        torchaudio.save('{}/audios/recon_{}.wav'.format(result_path, i), recon_audio,
                        data[0][1].item())
        print(recon_audio.shape, data[0][0].shape)
        show_watermark(watermark, wm_decode_att, '{}/audios/wm_{}.jpg'.format(result_path, i))

# test(model_restoration, wm_ds)
train(wm_ds, train_dl)
# fine_tune(model_restoration, wm_ds)