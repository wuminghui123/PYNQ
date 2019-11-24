from pynq.overlays.base import BaseOverlay
from pynq.lib import Pmod_OLED
base_overlay = BaseOverlay("base.bit")
pmod_oled = Pmod_OLED(base_overlay.PMODA)
from pynq import Overlay
import wave
pAudio = base_overlay.audio
pAudio.select_microphone()
pAudio.bypass(5)
def Wait():
    while base_overlay.buttons[0].read()==0:
        pass
def record(name):
    pAudio.record(5)
    pAudio.save(name + '.wav')
def vedio(name):
    pAudio.load(name + '.wav')
    pAudio.play()
from train import train
from test import test
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fftpack import fft
def main():
    pmod_oled.clear()
    pmod_oled.write('First\nPress the button to start')
    #print('请选择词语设置为密码：输入三次，推荐密码为五位数字')
    #print('第一次：按回车开始')
    Wait()
    pmod_oled.clear()
    pmod_oled.write('Recording')
    record('1')
    #print('第一次记录完成')
    pmod_oled.clear()
    pmod_oled.write('Second\nPress the button to start')
    Wait()
    pmod_oled.clear()
    pmod_oled.write('Recording')
    record('2')
    #print('第二次完成')
    #print('第三次：按回车继续')
    pmod_oled.clear()
    pmod_oled.write('Third\nPress the button to start')
    Wait()
    pmod_oled.clear()
    pmod_oled.write('Recording')
    record('3')
    pmod_oled.clear()
    pmod_oled.write('Record Complete\nTraining')

    code , dkmax, dkmin = train(3)
    pmod_oled.write('Train Complete\nPress the button to test')
    while True:
        Wait()
        pmod_oled.clear()
        pmod_oled.write('Recording')
        record('test')
        pmod_oled.clear()
        pmod_oled.write('Testing')
        y=test('test', code)
        if y:
            pmod_oled.clear()
            pmod_oled.write('Correct\nPress the button to test again')
        else:
            pmod_oled.clear()
            pmod_oled.write('ERROR\nPress the button to test again')
main()