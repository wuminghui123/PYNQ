import numpy

def disteu(x, y):
    # print(x.shape, y.shape)
    try:
        M, N = x.shape
    except ValueError:
        M = x.shape[0]
        N = 1
    try:
        M2, P = y.shape
    except ValueError:
        M2 = y.shape[0]
        P = 1

    # print(M, N, M2, P)

    if M != M2:
        print('矩阵维数不匹配')
        exit()


    d = numpy.zeros((N, P))
    # print(d)

    if N < P:
        for i in range(N):
            q = x[:, i]
            for _ in range(P - 1):
                q = numpy.column_stack(q, x[:, i])
            d[i, :] = sum(q - y) ** 2
    else:
        for i in range(P):
            try:
                q = y[:, i]
                for _ in range(N - 1):
                    q = numpy.column_stack((q, y[:, i]))
                d[:, i] = sum((x - q) ** 2).T
            except IndexError:
                q = y
                for _ in range(N - 1):
                    q = numpy.column_stack((q, y))
                # print(q)
                d[:, i] = sum((x - q) ** 2).T

    return d ** 0.5