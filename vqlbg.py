import numpy
from disteu import disteu

def vqlbg(c, k):
    e = 0.01
    r = numpy.mean(c, 1)
    dpr = 10000

    for i in range(int(numpy.log2(k))):
        r = numpy.column_stack((r * (1 + e), r * (1 - e)))
        while True:
            z = disteu(c, r)
            # print(z)
            m = z.min(axis=1)
            ind = z.argmin(axis=1)
            # print(ind)

            t = 0
            for j in range(2 ** (i+1)):
                try:
                    r[:, j] = numpy.mean(numpy.squeeze(c[:, numpy.where(ind == j)]), 1)
                except IndexError:
                    r[:, j] = numpy.squeeze(c[:, numpy.where(ind == j)])
                x = disteu(numpy.squeeze(c[:, numpy.where(ind == j)]), r[:, j])
                for q in range(len(x)):
                    t = t + x[q]
                    # print(t)
            if ((dpr - t) / t) < e:
                break
            else:
                # print((dpr - t)/t, t)
                dpr = t
    return r