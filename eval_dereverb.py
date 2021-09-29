"""
Evaluate WER with speech brain pretained transformer model
Memory cost a lot, extreme long speech will cause OOM
Current API does not support to use gpu easily. Change the source code and reinstall
"""
import sys
import argparse
import csv
from collections import defaultdict
from itertools import combinations, product
import logging
import os
import glob
import pickle
import json
import time
import random
import shutil
import warnings

import torch
import numpy as np
import librosa
import multiprocessing as mp
import matplotlib.pyplot as plt
from pesq import pesq
import speechmetrics
import torchaudio
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from speechbrain.utils.metric_stats import ErrorRateStats, EER, minDCF
from speechbrain.pretrained import TransformerASR, SpeakerRecognition
import torchvision
from torch.utils.tensorboard import SummaryWriter

from soundspaces.tasks.nav import compute_spectrogram
from lhs.predictor import Predictor
from ss_baselines.common.utils import to_tensor
from lhs.dataset import normalize
from lhs.room_acoustics_utils import measure_rt60
from lhs.trainer import griffinlim
from lhs.render_human import SPEAKER_SEMANTIC_ID

from librosa.util import normalize
from scipy.io.wavfile import write
from dataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from generator import Generator
from utils import HParam

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

EPS = 1e-7
device = torch.device("cuda", 0)
parser = argparse.ArgumentParser("Evaluate WER of dataset")
parser.add_argument('--data-list', type=str, default="data/sounds/speech/LibriSpeech/test-clean.csv",
                    help='list of eval data')
parser.add_argument('--enhanced-dir', type=str, default="data/sounds/speech/test_enhanced_wavs",
                    help='path to specific wavfolder ')
parser.add_argument('--print-frequency', type=int, default=10,
                    help='the frequency to print current WER')
parser.add_argument('--split', type=str, default='test-unseen', choices=['train', 'test-unseen', 'val', 'val-mini',
                                                                         'test-seen', 'test-unseen-v2',
                                                                         'test-unseen-v3', 'test-unseen-no-human',
                                                                         'test-unseen-qual', 'test-real'],
                    help='the path to precomputed pickle files')
parser.add_argument('--use-rgb', default=False, action='store_true',
                    help='whether use visual information')
parser.add_argument('--use-depth', default=False, action='store_true',
                    help='whether use visual information')
parser.add_argument('--use-rgbd', default=False, action='store_true',
                    help='whether use visual information')
parser.add_argument('--use-seg', default=False, action='store_true',
                    help='whether use visual information')
parser.add_argument('--use-location', default=False, action='store_true',
                    help='whether use location information')
parser.add_argument("--use-noise", default=False, action='store_true',
                    help="Modify config options from command line", )
parser.add_argument("--snr", default=20, type=float,
                    help="snr when adding noise")
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                    help='whether use location information')
parser.add_argument('--model-dir', type=str, default='',
                    help='whether use location information')
parser.add_argument('--pretrained-path', type=str, default='',
                    help='whether use location information')
parser.add_argument('--ckpt', type=int, default=-1,
                    help='whether use location information')
parser.add_argument('--stats', type=str, default='_stats',
                    help='suffix for the test json files')
parser.add_argument('--num-channel', type=int, default=1,
                    help='if use real ratio, channel is 1;  if complex, channel is 2')
parser.add_argument('--gpu-id', type=int, default=0,
                    help='assign files to gpu')
parser.add_argument('--gpu-count', type=int, default=1,
                    help='assign files to gpu')
parser.add_argument('--example-path', type=str, default='data/examples',
                    help='compute WER for predictions or raw reverb ')
parser.add_argument('--draw', default=False, action='store_true', help='whether to draw some features')
parser.add_argument('--est-pred', default=False, action='store_true',
                    help='compute WER for predictions or raw reverb ')
parser.add_argument('--use-clean', default=False, action='store_true',
                    help='compute WER for predictions or raw reverb ')
parser.add_argument('--eval-dereverb', default=False, action='store_true',
                    help='evaluate dereverberation performance')
parser.add_argument('--eval-asr', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--eval-spkrec', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--log-mag', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--log1p', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--norm-spec', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--use-real-imag', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--no-mask', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--save-audio', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--use-rad-loss', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--use-detection-loss', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--use-distance-loss', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--slurm', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--watch', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--limited-fov', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--crop', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--parallel', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--prev-ckpt-ind', default=-1, type=int,
                    help='evaluate ASR performance')
parser.add_argument('--max-ckpt-ind', default=150, type=int,
                    help='evaluate ASR performance')
parser.add_argument('--num-node', default=-1, type=int,
                    help='evaluate ASR performance')
parser.add_argument('--max-concurrent-job', default=40, type=int,
                    help='evaluate ASR performance')
parser.add_argument('--eval-interval', default=1, type=int,
                    help='evaluate ASR performance')
parser.add_argument('--overwrite', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--normalize', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--mean-pool-visual', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--test-all', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--use-triplet-fc', default=False, action='store_true',
                    help='evaluate ASR performance')
parser.add_argument('--partition', default='learnfair', type=str,
                    help='evaluate ASR performance')
parser.add_argument('--gpu-mem32', default=False, action='store_true',
                    help='evaluate ASR performance')
logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")


def load_model(args, checkpoint_path):
    if args.est_pred:
        checkpoint_file  = 'data/hifigan2/v3/chkpt/g_00240000'
        dereverber = Generator(1).to(args.device)
        state_dict_g = torch.load(checkpoint_file, map_location=args.device)
        dereverber.load_state_dict(state_dict_g['generator'], torch.device(args.device))
        dereverber.eval()
    else:
        dereverber = None

    if args.eval_asr:
        asr_model = TransformerASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech",
                                                savedir="pretrained_models/asr-transformer-transformerlm-librispeech")
    else:
        asr_model = None

    if args.eval_spkrec:
        verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                       savedir="pretrained_models/spkrec-ecapa-voxceleb")
    else:
        verification = None

    return dereverber, asr_model, verification


def overlap_chunk(input, dimension, size, step, left_padding):
    """
    Input shape is [Frequency bins, Frame numbers]
    """
    input = F.pad(input, (left_padding, size), 'constant', 0)
    return input.unfold(dimension, size, step)


def dereverberate(dereverber, input_data, args):
    """
        dereverb the data, return the waveform or spectrogram
    """
    tgt_shape = (256, 256)
    wav = input_data['receiver_audio']
    
    wav = wav / MAX_WAV_VALUE
    wav = normalize(wav) * 0.95
    wav = torch.FloatTensor(wav).to(device=device)
    wav = wav.reshape((1, 1, wav.shape[0],)).to(device)
    before_y_g_hat, y_g_hat = dereverber(wav, False)
    audio = before_y_g_hat.reshape((before_y_g_hat.shape[2],)).detach()
    # librosa.output.write_wav('test.wav', torchaudio.transforms.Resample(22050, 16000)(audio).cpu().numpy(), 16000)
    # exit(0)
    audio = audio * MAX_WAV_VALUE

    power_spec = None
    pred_audio = audio.unsqueeze(0)

    return power_spec, pred_audio


def parse_csv(args):
    test_list = {}
    with open(args.data_list) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        count = 0
        for row in csv_reader:
            if count > 0:
                id, dur, path, target = row[0], row[1], row[2], row[4].strip()
                test_list[id] = (path, target, float(dur))
            count += 1
    return test_list


def poll_checkpoint_folder(
        checkpoint_folder: str, previous_ckpt_ind: int, eval_interval: int, max_ckpt_ind=1000
):
    r""" Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
    (sorted by time of last modification).
    """
    assert os.path.isdir(checkpoint_folder), (
        f"invalid checkpoint folder " f"path {checkpoint_folder}"
    )
    models_paths = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/ckpt_*.pth"))
    )
    models_paths.sort(key=os.path.getmtime)
    # models_paths = [x for x in models_paths if x.split('/')[-1].startswith('ckpt_')]
    ind = previous_ckpt_ind + eval_interval
    if ind < len(models_paths) and ind < max_ckpt_ind:
        return models_paths[ind]
    else:
        return None


def eval_batch_continuous(args, files):
    if not args.watch:
        # by default evaluate the best checkpoint
        if args.est_pred:
            if args.ckpt != -1:
                checkpoint_path = os.path.join(args.model_dir, f'ckpt_{args.ckpt}.pth')
                # assert os.path.exists(checkpoint_path)
            elif args.pretrained_path != '':
                checkpoint_path = args.pretrained_path
            else:
                assert args.model_dir != ''
                if args.split == 'test-seen':
                    checkpoint_path = os.path.join(args.model_dir, 'best_val-seen.pth')
                else:
                    checkpoint_path = os.path.join(args.model_dir, 'best_val.pth')
        else:
            checkpoint_path = None
        eval_batch(args, files, checkpoint_path)
    else:
        writer = SummaryWriter(log_dir=os.path.join(args.model_dir, 'tb'))
        prev_ckpt_ind = args.prev_ckpt_ind
        while True:
            current_ckpt = None
            while current_ckpt is None:
                # in case the evaluation stops, specify the last evaluated checkpoint
                current_ckpt = poll_checkpoint_folder(
                    args.model_dir, prev_ckpt_ind, args.eval_interval
                )
                time.sleep(2)  # sleep for 2 secs before polling again
            logging.info(f"=======current_ckpt: {current_ckpt}=======")
            prev_ckpt_ind += args.eval_interval
            eval_batch(args, files,
                       checkpoint_path=current_ckpt,
                       writer=writer,
                       current_ckpt_ind=prev_ckpt_ind
                       )


def eval_batch(args, files, checkpoint_path, writer=None, current_ckpt_ind=0):
    """
    read the PKL file, perform dereveber and ASR
    """
    dereverber, asr_model, verification = load_model(args, checkpoint_path)
    stats_dir = os.path.join(args.model_dir, args.split + args.stats)
    metrics = speechmetrics.load(['stoi'])

    running_metrics = defaultdict(list)
    cer_stats = ErrorRateStats()
    test_id_list = parse_csv(args)
    wav_len = torch.tensor([1.0])
    sr = 16000
    count = 0
    if args.eval_spkrec:
        enhance_cache = {}
        pos_pairs, neg_pairs = load_spkcfg(files, args)
        print(f"num pos pairs:{len(pos_pairs)}, num neg pairs:{len(neg_pairs)}")

    if args.use_noise:
        noise_pool = load_noise(args)

    if args.save_audio:
        csv_lines = [["ID", "duration", "wav", "spk_id", "wrd"]]

    for file in tqdm(files):
        speech_id = os.path.basename(file)
        speech_id = speech_id.replace('.pkl', '')
        scene_id = file.split('/')[-2]
        non_duplicate_speech_id = f'{speech_id}_{scene_id}'

        count += 1
        with open(file, 'rb') as f:
            input_data = pickle.load(f)
        _, target, _ = test_id_list[speech_id]
        source_audio = to_tensor(input_data['source_audio'])
        receiver_audio = to_tensor(input_data['receiver_audio'])

        # from PIL import Image
        # rgb = np.array(Image.open('data/classroom_3.png'))
        # depth = np.load('data/classroom_2_disp.npy').squeeze(0)
        # depth = np.clip(depth, 0, 10) / 10
        # depth = torchvision.transforms.Resize((384, 1152))(to_tensor(depth))
        # input_data['rgb'] = rgb
        # input_data['depth'] = depth
        # import librosa
        # y, sr = librosa.load('data/trimmed_class_example.wav', sr=16000)
        # source_audio = to_tensor(y)
        # receiver_audio = to_tensor(y)

        if args.use_noise:
            waveform_length = receiver_audio.shape[0]
            noise_start_index = np.random.randint(0, noise_pool.shape[0] - waveform_length)
            noise = noise_pool[noise_start_index: noise_start_index + waveform_length]
            noise_energy = 10 * torch.log10(torch.sum(noise ** 2))
            signal_energy = 10 * torch.log10(torch.sum(receiver_audio ** 2))
            weight = torch.pow(10, ((signal_energy - args.snr) - noise_energy) / 20)
            receiver_audio += noise * weight

        stats = {}
        if args.est_pred:
            pred_spec, enhanced_audio = dereverberate(dereverber, input_data, args)
        else:
            if args.use_clean:
                enhanced_audio = source_audio.unsqueeze(0)
            else:
                # use reverberant audio
                enhanced_audio = receiver_audio.unsqueeze(0)
            pred_spec = torch.stft(enhanced_audio[0], n_fft=512, hop_length=160, win_length=400,
                                   window=torch.hamming_window(400), pad_mode='constant'). \
                pow(2).sum(-1).unsqueeze(0).to(device)

        if args.eval_spkrec:
            enhance_cache[non_duplicate_speech_id] = [source_audio.cpu(), enhanced_audio.cpu()]

        if args.eval_dereverb:
            reference = source_audio.numpy()
            enhanced = enhanced_audio.cpu()[0].numpy()
            stoi_score = metrics(enhanced, reference, rate=16000)['stoi'][0]
            pesq_score = pesq(16000, reference, enhanced, 'wb')
            running_metrics['stoi'].append(stoi_score)
            running_metrics['pesq'].append(pesq_score)
            stats['stoi'] = stoi_score
            stats['pesq'] = pesq_score

        if args.eval_asr:
            if args.num_channel == 1:
                pred, tokens = asr_model.transcribe_batch_spectrogram(pred_spec, wav_len)
            else:
                pred, tokens = asr_model.transcribe_batch(enhanced_audio, wav_len)
            stats['predict'] = pred[0]
            stats['target'] = target

            cer_stats.append(ids=[speech_id], predict=np.array([pred[0].split(' ')]),
                             target=np.array([target.split(' ')]))
            if count % args.print_frequency == 0:
                print(f"Current WER for first {count} sentences:", cer_stats.summarize()['WER'])

        if writer is None:
            with open(os.path.join(stats_dir, f'{non_duplicate_speech_id}.json'), 'w') as outfile:
                json.dump(stats, outfile)

        if args.draw:
            path = os.path.join(args.model_dir, args.split+'-supp', speech_id)
            os.makedirs(path, exist_ok=True)
            rec, src = normalize(receiver_audio, norm='peak'), normalize(source_audio, norm='peak')
            torchaudio.save(os.path.join(path, 'reverb.wav'), rec.unsqueeze(0), sr)
            torchaudio.save(os.path.join(path, 'source.wav'), src.unsqueeze(0), sr)
            torchaudio.save(os.path.join(path, 'pred.wav'), normalize(enhanced_audio.cpu()), sr)

            np.save(os.path.join(path,'recv_spec.npy'), torch.log1p(compute_spectrogram(rec, log=False, use_mag=True))[:, :, 0].numpy())
            np.save(os.path.join(path,'src_spec.npy'), torch.log1p(compute_spectrogram(src, log=False, use_mag=True))[:, :, 0].numpy())
            np.save(os.path.join(path,'pred_spec.npy'), torch.log1p(pred_spec.cpu().squeeze()).numpy())
            if isinstance(input_data['rgb'], list):
                np.save(os.path.join(path,'rgb.npy'),  np.concatenate([x / 255.0 for x in input_data['rgb']], axis=1))
            else:
                np.save(os.path.join(path, 'rgb.npy'), input_data['rgb'].numpy() / 255.0)

            # from librosa.display import specshow
            # plt.figure()
            #
            # specshow(torch.log1p(compute_spectrogram(rec, log=False, use_mag=True))[:, :, 0].numpy(), sr=16000)
            #
            # plt.savefig(os.path.join(path, 'recv_spec.png'), bbox_inches='tight', pad_inches=0)
            # plt.figure()
            # specshow(torch.log1p(compute_spectrogram(src, log=False, use_mag=True))[:, :, 0].numpy(), sr=16000)
            # plt.savefig(os.path.join(path, 'src_spec.png'), bbox_inches='tight', pad_inches=0)
            # plt.figure()
            # specshow(torch.log1p(pred_spec.cpu().squeeze()).numpy(), sr=16000)
            # plt.savefig(os.path.join(path, 'pred_spec.png'), bbox_inches='tight', pad_inches=0)
            #
            # plt.imsave(os.path.join(path, f'rgb.jpg'), np.concatenate([x / 255.0 for x in input_data['rgb']], axis=1))
            # dist = input_data['geodesic_distance']

            # with open(os.path.join(path, f'stat.json'), 'w') as fo:
            #     json.dump(fo, dict(dist=dist, target=target, pred=pred[0]))

        if args.save_audio:
            # save the audio and generate csv for finetuning
            # audio_dir = os.path.join(args.model_dir, args.split, str(input_data['geodesic_distance']) + non_duplicate_speech_id)
            audio_dir = os.path.join(args.model_dir, args.split, non_duplicate_speech_id)
            # Appending current file to the csv_lines list

            os.makedirs(audio_dir, exist_ok=True)
            if True:  # len(os.listdir(os.path.join(args.model_dir, args.split))) < 100:
                pred_file = os.path.join(audio_dir, f'pred.wav')
                duration = enhanced_audio.size(1) / sr
                spk_id = '-'.join(speech_id.split('-')[:2])

                torchaudio.save(pred_file, normalize(enhanced_audio.cpu()), sr)
                # torchaudio.save(os.path.join(audio_dir, 'src.wav'), normalize(source_audio.unsqueeze(0)), sr)
                # torchaudio.save(os.path.join(audio_dir, 'receiver.wav'), normalize(receiver_audio.unsqueeze(0)), sr)
                csv_line = [non_duplicate_speech_id, str(duration), pred_file, spk_id, target]
                csv_lines.append(csv_line)

    if args.eval_spkrec:
        negative_scores = compute_scores(neg_pairs, verification, enhance_cache).cpu()
        positive_scores = compute_scores(pos_pairs, verification, enhance_cache).cpu()

        # Final EER computation
        eer, th = EER(positive_scores, negative_scores)
        min_dcf, th = minDCF(positive_scores, negative_scores)
        print(f"EER:{eer}, min_dcf: {min_dcf}")
        with open(os.path.join(args.model_dir, args.split + "_spkrec_scores.pkl"), 'wb') as fo:
            pickle.dump({'pos': positive_scores, 'neg': negative_scores, "EER": eer, "min_dcf": min_dcf}, fo)

    if args.save_audio:
        with open(os.path.join(args.model_dir, args.split + '.csv'), mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_lines:
                csv_writer.writerow(line)

    metrics = {}
    if args.eval_dereverb:
        for metric, values in running_metrics.items():
            avg_metric_value = np.mean(values)
            print(metric, avg_metric_value)
            if writer is not None:
                writer.add_scalar(f'{args.split}/{metric}', avg_metric_value, current_ckpt_ind)
            metrics[metric] = avg_metric_value
    if args.eval_asr:
        wer = cer_stats.summarize()['WER']
        print(f"Final WER:", wer)
        if writer is not None:
            writer.add_scalar(f'{args.split}/wer', wer, current_ckpt_ind)
        metrics['wer'] = wer
    if args.eval_spkrec:
        if writer is not None:
            writer.add_scalar(f'{args.split}/eer', eer, current_ckpt_ind)
            writer.add_scalar(f'{args.split}/min_dcf', min_dcf, current_ckpt_ind)
        metrics['eer'] = eer
        metrics['min_dcf'] = min_dcf

    return metrics


def compute_scores(pairs, verification, enhance_cache):
    size = len(pairs)
    batch = 16
    scores = []
    for i in tqdm(range(0, size, batch)):
        s1, s2 = [], []
        wav_len1, wav_len2 = [], []
        for pair in pairs[i:i + batch]:
            key_1, key_2 = pair
            s1.append(enhance_cache[key_1][0].cuda().squeeze(0))
            s2.append(enhance_cache[key_2][1].cuda().squeeze(0))
            wav_len1.append(s1[-1].size(0))
            wav_len2.append(s2[-1].size(0))

        wav_len1, wav_len2 = torch.tensor(wav_len1).cuda().float(), torch.tensor(wav_len2).cuda().float()
        wav_len1, wav_len2 = wav_len1 / torch.max(wav_len1), wav_len2 / torch.max(wav_len2)

        ref = pad_sequence(s1, batch_first=True)
        compare = pad_sequence(s2, batch_first=True)
        score, prediction = verification.verify_batch(ref, compare, wav_len1, wav_len2)
        scores.append(score)
    return torch.cat(scores).squeeze()


def eval_metrics(args):
    """
    Eval the generated enhanced wavefiles.
    It has two modes. all test samples in the folder or all test samples in the test-clean list.
    """
    dereverber, asr_model, verification = load_model(args, None)

    metrics = speechmetrics.load(['stoi'])
    cer_stats = ErrorRateStats()
    test_dict = parse_csv(args)
    files = sorted(glob.glob(args.enhanced_dir + '/*.wav', recursive=True))
    num_metric_point = 0
    wav_len = torch.tensor([1.0])

    # slice the files into different chunks
    test_dir = os.path.join(args.enhanced_dir, "0")
    os.makedirs(test_dir, exist_ok=True)

    # files = distributing_files(files, args, mode='equal')

    meta_results = defaultdict(list)
    pesqs = []
    stois = []
    for wav_file in tqdm(files):
        id = os.path.basename(wav_file).split('-', 1)[1]
        # parse the file name and get the id
        id = id.replace('.wav', '').replace('.pkl', '')
        if id not in test_dict:
            continue
        source_path, target_text, _ = test_dict[id]
        source_audio, _ = torchaudio.load(source_path)

        predicted_audio, _ = torchaudio.load(wav_file)
        pred, tokens = asr_model.transcribe_batch(predicted_audio, wav_len)

        predicted_audio, source_audio = predicted_audio.squeeze(0).numpy(), source_audio.squeeze(0).numpy()
        scores = metrics(predicted_audio, source_audio, rate=16000)

        meta_results[id].append(scores['stoi'][0])
        stois.append(meta_results[id][-1])
        meta_results[id].append(pesq(16000, source_audio, predicted_audio, 'wb'))
        pesqs.append(meta_results[id][-1])

        meta_results[id].append(pred[0])
        meta_results[id].append(target_text)
        cer_stats.append(ids=[id], predict=np.array([pred[0].split(' ')]),
                         target=np.array([target_text.split(' ')]))

        num_metric_point += 1
        if num_metric_point % args.print_frequency == 0:
            logging.info(f"Current WER for first {num_metric_point} sentences:{cer_stats.summarize()['WER']}")

    logging.info(f"WER for all sentences: {cer_stats.summarize()['WER']}")
    logging.info(f"PESQ: {np.mean(np.array(pesqs))}, STOI: {np.mean(np.array(stois))}")
    path = os.path.join(test_dir, f"{args.gpu_id}.pkl")
    with open(path, 'wb') as fo:
        pickle.dump(meta_results, fo)
    logging.info(f"GPU {args.gpu_id} finalized test")


def load_spkcfg(files, args):
    spkrec_config = os.path.join('data/lhs_v2', args.split, 'spkrec_config.ark')
    # load id => sex
    with open("data/sounds/speech/LibriSpeech/SPEAKERS.TXT") as f:
        content = f.readlines()
    id2sex = {}
    for line in content[12:]:
        detail = line.split('|')
        id = detail[0].strip()
        sex = detail[1].strip()
        id2sex[id] = sex

    if os.path.exists(spkrec_config):
        with open(spkrec_config, 'rb') as fo:
            data = pickle.load(fo)
        return data['pos'], data['neg']
    else:
        spkid2file = defaultdict(list)
        non_duplicate_speech_ids = []
        file2sex = {}
        for file in files:
            spkid = file.split('/')[-1].split('-')[0]
            scene_id = file.split('/')[-2]
            speech_id = file.split('/')[-1].split('.')[0]
            non_duplicate_speech_id = f'{speech_id}_{scene_id}'
            spkid2file[spkid].append(non_duplicate_speech_id)
            non_duplicate_speech_ids.append(non_duplicate_speech_id)
            file2sex[non_duplicate_speech_id] = id2sex[spkid]
        pos_pairs, neg_pairs = list(), list()
        for k, v in spkid2file.items():
            pos_pairs += combinations(v, r=2)
            neg_pairs += product(v, [x for x in non_duplicate_speech_ids if not x.startswith(k)])
        num_sample = 1000 if args.split == 'val-mini' else 40000
        random.shuffle(pos_pairs)
        random.shuffle(neg_pairs)
        pos_pairs = pos_pairs[:num_sample]
        selected_neg_pairs = []
        count = 0
        for pair in neg_pairs:
            speech_0, speech_1 = pair
            if file2sex[speech_0] != file2sex[speech_1]:
                count += 1
                selected_neg_pairs.append(pair)
            if count >= num_sample:
                break
        # neg_pairs = neg_pairs[:num_sample]
        with open(spkrec_config, 'wb') as fo:
            pickle.dump({'pos': pos_pairs, 'neg': selected_neg_pairs}, fo)
        return pos_pairs, selected_neg_pairs


def load_noise(args):
    splitmap = {"train": "tr", "val": "cv", "val-mini": "cv", "test-unseen": "tt", "test-seen": "tt"}
    noise_cache_path = os.path.join('data/sounds/speech/wham_noise', splitmap[args.split], "noise.pkl")
    if os.path.exists(noise_cache_path):
        with open(noise_cache_path, 'rb') as fo:
            data = pickle.load(fo)
        noise_pool = data['noise']
    else:
        noise_files = sorted(glob.glob(os.path.join('data/sounds/speech/wham_noise', splitmap[args.split], "*.wav")))
        noise_files = np.random.choice(noise_files, 600)
        noise_list = []
        for noise_file in tqdm(noise_files):
            noise, _ = torchaudio.load(noise_file)
            noise_list.append(noise)
        noise_pool = torch.cat(noise_list, dim=1)[0, :]
        with open(noise_cache_path, 'wb') as fo:
            pickle.dump({'noise': noise_pool}, fo)
    return noise_pool


def eval_metrics_multiprocessing(args):
    test_dict = parse_csv(args)
    files = glob.glob(args.enhanced_dir + '/*.wav', recursive=True)
    pool = mp.Pool(processes=64)
    results = pool.map(compute_metrics, zip(files, [test_dict] * len(files)))

    avg_metricss = np.mean(np.array(results), axis=0)
    print(avg_metricss)


def compute_metrics(wav_file, test_dict):
    metrics = speechmetrics.load(['stoi'])
    id = os.path.basename(wav_file)
    # parse the file name and get the id
    id = id.replace('.wav', '').split('-', 1)[1]

    source_path, target_text = test_dict[id]
    predicted_audio, _ = torchaudio.load(wav_file)
    source_audio, _ = torchaudio.load(source_path)
    predicted_audio, source_audio = predicted_audio.squeeze(0).numpy(), source_audio.squeeze(0).numpy()
    scores = metrics(predicted_audio, source_audio, rate=16000)

    return scores['stoi'][0], pesq(16000, source_audio, predicted_audio, 'wb')


def distributing_files(files, args, mode='equal'):
    '''
    load balancing for bunch of files
    '''
    logging.info(f"total {len(files)} samples, gpu id:{args.gpu_id} gpu count:{args.gpu_count}")
    test_dict = parse_csv(args)
    wav_to_len = {}
    for wav_file in files:
        id = os.path.basename(wav_file)
        id = id.replace('.pkl', '').replace('.wav', '')
        if id not in test_dict:
            continue
        wav_to_len[wav_file] = test_dict[id][-1]
    files = [k for k, v in sorted(wav_to_len.items(), key=lambda item: item[1])]
    if mode == 'equal':
        # resort by duration, resample smoothly
        files = files[args.gpu_id:: args.gpu_count]
    elif mode == 'customize':
        '''
        manually change the load balance by current distribution
        '''
        total = len(files)
        slice_ind = int(0.45 * total)
        if args.gpu_id < 2:
            files = files[:slice_ind]
            files = files[args.gpu_id:: 2]
        else:
            files = files[slice_ind:]
            files = files[(args.gpu_id - 2):: (args.gpu_count - 2)]


def distributing_files(files, args, mode='equal'):
    '''
    load balancing for bunch of files
    '''
    logging.info(f"total {len(files)} samples, gpu id:{args.gpu_id} gpu count:{args.gpu_count}")
    test_dict = parse_csv(args)
    wav_to_len = {}
    for wav_file in files:
        id = os.path.basename(wav_file)
        id = id.replace('.pkl', '').replace('.wav', '')
        if id not in test_dict:
            continue
        wav_to_len[wav_file] = test_dict[id][-1]
    files = [k for k, v in sorted(wav_to_len.items(), key=lambda item: item[1])]
    if mode == 'equal':
        # resort by duration, resample smoothly
        ind = args.gpu_id % args.gpu_count
        files = files[args.gpu_id:: args.gpu_count]

    logging.info(f"eval {len(files)} samples in GPU {args.gpu_id}")
    return files


def main():
    args = parser.parse_args()
    if args.test_all:
        from lhs.find_best_ckpt import find_best_ckpt_idx

        # if args.est_pred:
        #     best_indices = find_best_ckpt_idx(args.model_dir)
        # else:
        best_indices = {'pesq': 0, 'wer': 0, 'eer': 0}
        args.eval_dereverb = True
        args.eval_asr = False
        args.eval_spkrec = False
        args.ckpt = best_indices['pesq']
        args.num_node = 1
        args.stats = '_stats'
        test(args)

        args.eval_dereverb = False
        args.eval_asr = True
        args.eval_spkrec = False
        args.ckpt = best_indices['wer']
        args.num_node = 4
        args.stats = '_stats'
        test(args)

        args.eval_dereverb = False
        args.eval_asr = False
        args.eval_spkrec = True
        args.ckpt = best_indices['eer']
        args.num_node = 1
        args.stats = '_stats'
        test(args)
    else:
        test(args)


def test(args):
    if args.split == 'train':
        args.data_list = 'data/sounds/speech/LibriSpeech/train-360.csv'
    elif args.split.startswith('val'):
        args.data_list = 'data/sounds/speech/LibriSpeech/dev-clean.csv'
    elif args.split.startswith('test'):
        args.data_list = 'data/sounds/speech/LibriSpeech/test-clean.csv'
    else:
        raise ValueError

    if sum([args.eval_dereverb, args.eval_asr, args.eval_spkrec]) == 1 and args.stats == '_stats':
        if args.eval_dereverb:
            args.stats = '_pesq' if not args.use_noise else '_noise_pesq'
        if args.eval_asr:
            args.stats = '_wer' if not args.use_noise else '_noise_wer'
        if args.eval_spkrec:
            args.stats = '_eer' if not args.use_noise else '_noise_eer'

    stats_dir = os.path.join(args.model_dir, args.split + args.stats)
    print(f'Stats dir: {stats_dir}')
    # if os.path.exists(stats_dir):
    #     if args.overwrite or input('Model dir exists. Overwrite?\n') == 'y':
    #         shutil.rmtree(stats_dir)
    os.makedirs(stats_dir, exist_ok=True)
    random.seed(0)
    np.random.seed(0)

    data_dir = os.path.join('data/lhs_v2', args.split)
    files = sorted(glob.glob(data_dir + '/**/*.pkl', recursive=True))
    assert len(files) != 0

    if args.slurm:
        if args.parallel:
            # evaluate multiple validation checkpoints at the same time
            import submitit
            from submitit.core.utils import FailedJobError, UncompletedJobError
            executor = submitit.AutoExecutor(folder="data/logs/submitit/%j")
            executor.update_parameters(job_name=args.model_dir.strip('/').split('/')[-1], timeout_min=70,
                                       partition=args.partition, gpus_per_node=1, cpus_per_task=10,
                                       constraint='volta32gb' if args.gpu_mem32 else None)
            jobs = []
            indices = []

            writer = SummaryWriter(log_dir=os.path.join(args.model_dir, 'tb'))
            prev_ckpt_ind = args.prev_ckpt_ind
            while True:
                current_ckpt = None
                while current_ckpt is None:
                    # in case the evaluation stops, specify the last evaluated checkpoint
                    current_ckpt = poll_checkpoint_folder(
                        args.model_dir, prev_ckpt_ind, args.eval_interval, args.max_ckpt_ind
                    )
                    time.sleep(2)  # sleep for 2 secs before polling again

                    # check if jobs have been finished
                    while True:
                        if len(jobs) > 0 and jobs[0].done():
                            try:
                                metrics = jobs[0].result()
                                for metric, value in metrics.items():
                                    writer.add_scalar(f'{args.split}/{metric}', value, indices[0])
                                logging.info(f"=======Finished evaluating {indices[0]} ckpt=======")
                            except (FailedJobError, UncompletedJobError):
                                logging.info(f"=======Failed evaluating {indices[0]} ckpt=======")
                            jobs.pop(0)
                            indices.pop(0)
                        elif len(jobs) >= args.max_concurrent_job:
                            time.sleep(2)
                        else:
                            break

                # submit the current job
                prev_ckpt_ind += args.eval_interval
                assert str(prev_ckpt_ind) in current_ckpt
                try:
                    job = executor.submit(eval_batch, args, files, current_ckpt, None, prev_ckpt_ind)
                    logging.info(f"=======current_ckpt: {current_ckpt}=======")
                    jobs.append(job)
                    indices.append(prev_ckpt_ind)
                except FailedJobError:
                    logging.info(f"Failed to submit job for {current_ckpt}")
        else:
            num_machine = 4 if args.num_node == -1 else args.num_node
            timout_min = 75
            step = int(len(files) / num_machine) + 1

            import submitit
            executor = submitit.AutoExecutor(folder="data/logs/submitit/%j")
            executor.update_parameters(job_name=args.model_dir.strip('/').split('/')[-1], timeout_min=timout_min,
                                       partition=args.partition, gpus_per_node=1, cpus_per_task=10,
                                       constraint='volta32gb' if args.gpu_mem32 else None)
            jobs = []
            with executor.batch():
                for i in range(num_machine):
                    job = executor.submit(eval_batch_continuous, args, files[i * step: (i + 1) * step])
                    jobs.append(job)
    else:
        eval_batch_continuous(args, files)


if __name__ == '__main__':
    main()
