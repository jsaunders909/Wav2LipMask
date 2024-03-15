from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models.unet import UNETMask
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
parser.add_argument('--verbose', action='store_true')


args = parser.parse_args()



def display_image(x, x_masked):
    # B, 3*T, W, H -> B, T, W, H, 3
    B, t3, w, h = x.shape
    T = t3//3

    x = x.reshape(B, T, 3, w, h).permute((0, 1, 3, 4, 2)).detach().cpu().numpy() * 255
    x_masked = x_masked.reshape(B, T, 3, w, h).permute((0, 1, 3, 4, 2)).detach().cpu().numpy() * 255

    x, x_masked = x[0], x_masked[0]

    vis_in = np.concatenate([x[0], x[1], x[2], x[3], x[4]], axis=0)
    vis_out = np.concatenate([x_masked[0], x_masked[1], x_masked[2], x_masked[3], x_masked[4]], axis=0)
    vis = np.concatenate([vis_in, vis_out], axis=1)

    cv2.imwrite('vis.png', vis)

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split, verbose=False):
        self.all_videos = get_image_list(args.data_root, split)
        self.verbose = verbose

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx: end_idx, :]


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                if self.verbose:
                    print('Not enough frames')
                    print(f'Looking in {vidname}, images = {img_names}')
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                if self.verbose:
                    print('Couldnt get window')
                    print(chosen)
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    print(f'Cant read image {fname}')
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    print(f'Cant resize image {fname}')
                    all_read = False
                    break

                window.append(img)

            if not all_read:
                if self.verbose:
                    print('Error loading images')
                continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
                #print(f'Wav shape = {wav.shape}')
                #print(f'Orig mel shape = {orig_mel.shape}')
            except Exception as e:
                if self.verbose:
                    print('Error loading audio')
                    print(f'Audio = {wavpath}')
                    print(e)
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                if self.verbose:
                    print('MEL Shape incorrect')
                    print(f'Should be {syncnet_mel_step_size} but is {mel.shape}')
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)

    for i in range(d.shape[0]):
        print(d[i].item(), y[i, 0].item())
    loss = logloss(d.unsqueeze(1), y)
    return loss


def certainty_loss(a, v):
    d = nn.functional.cosine_similarity(a, v).unsqueeze(1)
    y = 0.5 * torch.ones_like(d)
    loss = logloss(d, y)
    return loss

def train(device, syncnet, unet, train_data_loader, test_data_loader, optimizer,
          syncnet_optimizer, checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step

    TRAIN_MASK = 10

    syncnet.eval()
    unet.train()

    while global_epoch < nepochs:
        running_loss, running_loss_sync, running_loss_reg = 0, 0, 0
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel, y) in prog_bar:

            # Transform data to CUDA device
            x = x.to(device)
            mel = mel.to(device)
            y = y.to(device)

            B, t3, w, h = x.shape
            inputs_ind = x.reshape(B, t3 // 3, 3, w, h).reshape(-1, 3, w, h)
            mask = unet(inputs_ind).reshape(B, t3 // 3, 1, w, h).repeat(1, 1, 3, 1, 1).reshape(B, t3, w, h)
            x_masked = x * mask

            # --------------------- Train Syncnet to be confident
            syncnet_optimizer.zero_grad()

            a, v = syncnet(mel, x_masked.detach())
            loss = cosine_loss(a, v, y)
            loss.backward()
            running_loss_sync += loss.item()

            syncnet_optimizer.step()

            optimizer.zero_grad()
            if global_step % TRAIN_MASK == 0:

                a, v = syncnet(mel, x_masked)
                loss = certainty_loss(a, v)

                running_loss += loss.item()

                print(mask.mean().item(), mask.max().item(), mask.min().item())
                reg_loss = (1 - mask).mean()

                running_loss_reg += reg_loss.item()

                loss = loss + 0.01 * reg_loss

                loss.backward()

                # optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step

            if global_step == 1 or global_step % 1000 == 0:
                save_checkpoint(
                    unet, optimizer, global_step, checkpoint_dir, global_epoch)

            #if global_step % hparams.syncnet_eval_interval == 0:
            #    with torch.no_grad():
            #        eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

            if global_step % 100 == 0:
                display_image(x, x_masked)

            prog_bar.set_description(f'Loss: Uncertainty {running_loss / (step + 1):03E}, Sync {running_loss_sync / (step + 1):03E}), Reg {running_loss_reg / (step + 1)}')

            torch.cuda.empty_cache()

        global_epoch += 1

def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)

        return

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step_mask{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    train_dataset = Dataset('train', verbose=args.verbose)
    test_dataset = Dataset('val', verbose=args.verbose)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=8)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    syncnet = SyncNet().to(device)
    unet = UNETMask().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in unet.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in unet.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    syncnet_optimizer = optim.Adam([p for p in syncnet.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    checkpoint_path = './checkpoints/lipsync_expert2.pth'
    if not os.path.exists(checkpoint_path):
        raise ValueError('A loaded checkpoint is required for training')
    load_checkpoint(checkpoint_path, syncnet, optimizer, reset_optimizer=False)

    train(device, syncnet, unet, train_data_loader, test_data_loader, optimizer, syncnet_optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)
