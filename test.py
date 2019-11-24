import scipy.io.wavfile
from mfcc import mfcc
from disteu import disteu
import numpy as np
import cut
def test(name, code):
    import wave
    wav_file=wave.open(name + '.wav')
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
    
    #sample_rate = wav_file.getframerate()/3
    sample_rate = 16000
    signal=frames[::3,0]
    sample_rate, signal = cut.cut(sample_rate, signal)
    v = mfcc(signal, sample_rate)

    distmin = float('inf')
    dist = 0

    for i in range(len(code)):
        d = disteu(v, code[i])
        dist += sum(d.min(1)) / d.shape[0]
    dist = dist / len(code)
        # dist = sum(d.min(1)) / d.shape[0]
        #
        # if dist < distmin:
        #     distmin = dist

    print(dist)

    if dist < 4.0:
        return True
    else:
        return False

