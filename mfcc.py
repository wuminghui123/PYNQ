import numpy
import melfb
from scipy.fftpack import dct


def mfcc(s, rate):
    a = 0.98
    length = len(s)

    # s1 = [0 for i in range(length)]
    s1 = numpy.zeros(length)

    for i in range(1, length):
        s1[i] = s[i] - a * s[i - 1]

    n = 320
    m = 160

    frame = int(numpy.floor((length - n) / m) + 1)

    z = numpy.zeros((n, frame))

    for j in range(frame):
        for i in range(n):
            z[i][j] = s1[j * m + i]

    h = numpy.hamming(n)

    z2 = numpy.zeros((n, frame))
    for j in range(frame):
        for i in range(n):
            z2[i][j] = h[i] * z[i][j]

    FFT = numpy.zeros((n, frame), dtype=complex)
    for i in range(frame):
        FFT[:, i] = numpy.fft.fft(z2[:, i])

    fs = 16000
    m = melfb.melfb(20, n, fs)
    n2 = 1 + int(numpy.floor(n / 2))
    mel = m * abs(FFT[0:n2, :]) ** 2
    # print(mel)
    # print(mel.shape)

    c = dct(numpy.log(mel), type=2, axis=0,  norm='ortho')
    c = numpy.delete(c, 0, 0)

    return c