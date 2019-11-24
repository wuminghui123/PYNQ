import scipy.io.wavfile
import numpy
import scipy.signal

def cut(sample_rate,signal):
    h = numpy.hamming(320)
    signal = signal / (2 ** 15)
    # print(signal.shape)
    e = scipy.signal.convolve(signal ** 2, h)
    mx = max(e)
    n = len(e)

    signal.flags.writeable = True
    signal = numpy.append(signal, numpy.zeros(n - len(signal)))

    for i in range(n):
        if e[i] < mx * 0.01:
            e[i] = 0
        else:
            e[i] = 1
    y = signal * e

    signal = list(y)

    for i in range(len(signal) - 1, -1, -1):
        if signal[i] == 0:
            signal.pop(i)

    signal = numpy.array(signal)

    return sample_rate,signal