import numpy
from scipy.sparse import csr_matrix

def melfb(p, n, fs):
    f0 = 700 / fs
    fn2 = int(numpy.floor(n/2))

    lr = numpy.log(1+0.5/f0) / (p+1)
    # print(lr)

    bl = n * (f0 * (numpy.exp(numpy.array([0, 1, p, p+1]) * lr) - 1))
    # print(bl)

    b1 = int(numpy.floor(bl[0]) + 1)
    b2 = int(numpy.ceil(bl[1]))
    b3 = int(numpy.floor(bl[2]))
    b4 = min(fn2, numpy.ceil(bl[3])) - 1
    # print(b1, b2, b3, b4)

    pf = numpy.log(1 + (numpy.arange(b1, b4+1))/n/f0) / lr
    fp = numpy.floor(pf)
    pm = pf - fp
    # print(pm)
    # print(fp)

    # print(fp[0:int(b3-1)])
    r = numpy.append(fp[b2-1:b4] , 1+fp[0:b3]) - 1
    # print(r)
    c = numpy.append(numpy.arange(b2, b4+1), numpy.arange(1, b3+1))
    # print(c)
    v = 2 * numpy.append(1 - pm[b2-1:b4], pm[0:b3])
    # print(v)

    m = csr_matrix((v, (r, c)), shape=(p, 1+fn2))
    # print(m.shape)
    return m

# melfb(20, 320, 16000)