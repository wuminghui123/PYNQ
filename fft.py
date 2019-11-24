from math import*
from pynq import Overlay
from pynq import Xlnk
import numpy as np
def fft(ol,data_in):
    out = ol.axi_dma_out
    re = ol.axi_dma_re
    im = ol.axi_dma_im
    data_size = 512
    xlnk = Xlnk()
    input_buffer = xlnk.cma_array(shape=(data_size,), dtype=np.int32)
    output_buffer_re = xlnk.cma_array(shape=(data_size,), dtype=np.int32)
    output_buffer_im = xlnk.cma_array(shape=(data_size,), dtype=np.int32)
    for i in range(512):
        input_buffer[i]=data_in[i]
    out.sendchannel.transfer(input_buffer)
    re.recvchannel.transfer(output_buffer_re)
    im.recvchannel.transfer(output_buffer_im)
    data_re=np.zeros(512)
    data_im=np.zeros(512)
    FFT=np.zeros(512)
    for i in range(512):
        if output_buffer_im[i]>=0x4000000:
            data_im[i]=-(0x8000000-output_buffer_im[i])
        else:
            data_im[i]=output_buffer_im[i]
    for i in range(512):
        if output_buffer_re[i]>=0x4000000:
            data_re[i]=-(0x8000000-output_buffer_re[i])
        else:
            data_re[i]=output_buffer_re[i]
    FFT=data_re*data_re+data_im*data_im
    return FFT