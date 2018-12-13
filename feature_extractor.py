import argparse
from time import time
import os
from multiprocessing import Pool
import numpy as np
from pydub import AudioSegment
import scipy.signal

overlapping = 512
chunk_size = 1024
num_frames = 25


def calc_spectrum(data):
    target_len = 16000

    prev_len = len(data)
    if prev_len < target_len:
        trunc_samples = np.pad(data, (0, target_len - prev_len), 'constant', constant_values=0)
    else:
        trunc_samples = data[:target_len]
    #     print(trunc_samples.shape)

    freqs = np.zeros((num_frames, chunk_size // 2 + 1), dtype=np.complex_)
    for i in range(num_frames):
        chunk = trunc_samples[i * 512:i * 512 + chunk_size]
        freqs[i, :] = np.fft.rfft(chunk)
    return freqs


def extract_feature(filepath, savepath):
    tracks = AudioSegment.from_wav(filepath)
    monotracks = tracks.split_to_mono()
    all_freqs = np.zeros((tracks.channels, num_frames, chunk_size // 2 + 1), dtype=np.complex_)
    for i in range(tracks.channels):
        track = (np.array(monotracks[i].get_array_of_samples()) / tracks.max_possible_amplitude)
        target_rate = 16000
        expect_len = int(len(track) * target_rate / tracks.frame_rate)
        if expect_len != len(track):
            resampled = scipy.signal.resample(track, expect_len)
            print('resampled from {} to {}'.format(len(track), expect_len))
            all_freqs[i, :, :] = calc_spectrum(resampled)
        else:
            all_freqs[i, :, :] = calc_spectrum(track)
    squared = np.square(np.absolute(all_freqs))
    energy = squared[0, :] + (squared[1, :] + squared[2, :] + squared[3, :])/3
    WX = np.multiply(all_freqs[0, :, :], all_freqs[3, :, :])
    WY = np.multiply(all_freqs[0, :, :], all_freqs[1, :, :])
    WZ = np.multiply(all_freqs[0, :, :], all_freqs[2, :, :])
    intensity_vecs = np.zeros((num_frames, chunk_size // 2 + 1, 6), dtype='float32')
    intensity_vecs[:, :, 0] = WX.real / energy
    intensity_vecs[:, :, 1] = WY.real / energy
    intensity_vecs[:, :, 2] = WZ.real / energy
    intensity_vecs[:, :, 3] = WX.imag / energy
    intensity_vecs[:, :, 4] = WY.imag / energy
    intensity_vecs[:, :, 5] = WZ.imag / energy

    if np.isnan(intensity_vecs).any() or np.isinf(intensity_vecs).any():
        print('file {} not saved due to invalid values'.format(savepath))
        return
    else:
        np.save(savepath, intensity_vecs)


def main():
    parser = argparse.ArgumentParser(prog='feature_extractor',
                                     description="""Script to convert ambisonic audio to intensity vectors""")
    parser.add_argument("--audiodir", "-d", help="Directory where audio files are located",
                        type=str, required=True)
    parser.add_argument("--output", "-o", help="Directory where feature files are written to",
                        type=str, required=True)
    parser.add_argument("--nthreads", "-n", type=int, default=1, help="Number of threads to use")

    args = parser.parse_args()
    audiodir = args.audiodir
    nthreads = args.nthreads
    outpath = args.output

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    ts = time()
    # Convert all houses
    try:
        # # Create a pool to communicate with the worker threads
        pool = Pool(processes=nthreads)
        for subdir, dirs, files in os.walk(audiodir):
            for f in files:
                if f.endswith('.wav'):
                    filename = os.path.join(subdir, f)
                    savepath = filename.replace(audiodir, outpath).replace('.wav', '.npy')
                    pool.apply_async(extract_feature, args=(filename, savepath))
    except Exception as e:
        print(str(e))
        pool.close()
    pool.close()
    pool.join()
    print('Took {}'.format(time() - ts))


if __name__ == "__main__":
    main()
