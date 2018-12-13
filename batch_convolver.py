import argparse
import os
from time import time
from multiprocessing import Pool
import random
import pydub as pd
import numpy as np
import scipy.signal as ssi
import math


def convolve(irfile_path, speech_path, output_path, target_len=1):
    IR = pd.AudioSegment.from_file(irfile_path)
    speech = pd.AudioSegment.from_file(speech_path)

    tracks = IR.split_to_mono()
    speechsamples = np.array(speech.get_array_of_samples()) / speech.max_possible_amplitude
    if len(speechsamples) > speech.frame_rate * target_len:
        rand_start = random.randint(0, len(speechsamples) - speech.frame_rate * target_len - 1)
        speechsamples = speechsamples[rand_start:(rand_start + speech.frame_rate * target_len)]
    convolved = []
    for i in range(len(tracks)):
        IRsamples = np.array(tracks[i].get_array_of_samples()) / IR.max_possible_amplitude
        if IR.frame_rate != speech.frame_rate:
            newlen = int(math.ceil(len(IRsamples) * speech.frame_rate / IR.frame_rate))
            IRsamples = ssi.resample(IRsamples, newlen)
        temp = np.convolve(speechsamples, IRsamples)
        convolved.append(temp)
    convolved = np.array(convolved)
    maxval = np.max(np.fabs(convolved))
    if maxval == 0:
        print("file {} not saved due to zero strength".format(output_path))
        return -1
    amp_ratio = 1.0 / maxval
    convolved *= amp_ratio
    convolved *= IR.max_possible_amplitude
    rawdata = convolved.transpose().astype(np.int32).tobytes()
    sound = pd.AudioSegment(data=rawdata, sample_width=IR.sample_width, frame_rate=speech.frame_rate, channels=IR.channels)
    sound.export(output_path, format='wav')


def main():
    parser = argparse.ArgumentParser(prog='batch_colvolver',
                                     description="""Batch convolve IR folder with speech folder""")
    parser.add_argument("--irfolder", "-i", help="Directory containing IR files", type=str, required=True)
    parser.add_argument("--speechfolder", "-s", help="Directory containing speech clips", type=str, required=True)
    parser.add_argument("--output", "-o", help="Output directory", type=str, required=True)
    parser.add_argument("--nthreads", "-n", type=int, default=1, help="Number of threads to use")

    args = parser.parse_args()
    irpath = args.irfolder
    speechpath = args.speechfolder
    nthreads = args.nthreads
    outpath = args.output

    if not os.path.exists(irpath):
        print('IR folder {} non-exist, abort!'.format(irpath))
        return

    if not os.path.exists(speechpath):
        print('Speech folder {} non-exist, abort!'.format(speechpath))
        return

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    irlist = [os.path.join(root, name) for root, dirs, files in os.walk(irpath)
              for name in files if name.endswith(".wav")]
    speechlist = [os.path.join(root, name) for root, dirs, files in os.walk(speechpath)
              for name in files if name.endswith((".wav", ".flac"))]

    ts = time()
    pool = Pool(processes=nthreads)
    res = []
    try:
        # Create a pool to communicate with the worker threads
        for irfile_path in irlist:
            output_path = irfile_path.replace(irpath, outpath)
            new_dir = os.path.dirname(output_path)
            speech_path = random.choice(speechlist)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            pool.apply_async(convolve, args=(irfile_path, speech_path, output_path,))
    except Exception as e:
        print(e)
        pool.close()
    pool.close()
    pool.join()

    print('Took {}'.format(time() - ts))


if __name__ == '__main__':
    main()