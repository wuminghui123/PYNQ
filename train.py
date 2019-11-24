import scipy.io.wavfile
from mfcc import mfcc
from vqlbg import vqlbg
from disteu import disteu
import numpy as np
import cut
def train(n):
    k = 16
    import wave
    code = []

    for i in range(n):
        wav_file=wave.open('{}'.format(i+1) + '.wav')
        raw_frames = wav_file.readframes(-1)
        num_frames = wav_file.getnframes()
        num_channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        temp_buffer = np.empty((num_frames, num_channels, 4), dtype=np.uint8)
        raw_bytes = np.frombuffer(raw_frames, dtype=np.uint8)
        temp_buffer[:, :, :sample_width] = raw_bytes.reshape(-1, num_channels, 
                                                    sample_width)
        temp_buffer[:, :, sample_width:] = \
            (temp_buffer[:, :, sample_width-1:sample_width] >> 7) * 255
        frames = temp_buffer.view('<i4').reshape(temp_buffer.shape[:-1])
        
        #sample_rate = wav_file.getframerate()
        sample_rate = 16000;
        signal=frames[::3,0]
        sample_rate, signal = cut.cut(sample_rate, signal)
#        sample_rate, signal = scipy.io.wavfile.read('{}'.format(i+1) + '.wav')
        v = mfcc(signal, sample_rate)

        code.append(vqlbg(v, k))
        print(i)


    if n == 3:
        d1 = disteu(code[0], code[1])
        d2 = disteu(code[1], code[2])
        d3 = disteu(code[2], code[0])

        dk = []
        dk.append(sum(d1.min(1)) / d1.shape[0])
        dk.append(sum(d2.min(1)) / d2.shape[0])
        dk.append(sum(d3.min(1)) / d3.shape[0])

        dmax = max(dk)
        dmin = min(dk)
    else:
        dmax = 0
        dmin = 0

    return code, dmax, dmin
