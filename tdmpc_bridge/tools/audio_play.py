# AudioPlay_playsound
# sudo apt install libgirepository1.0-dev
# pip install pygobject
# pip install playsound

# AudioPlay_pydub
# pip install pydub

import os
import time
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play

FILENAME = '/home/nuc/audio_test.m4a'

def AudioPlay_playsound(filename=FILENAME):
    playsound(filename)
    
def AudioPlay_pydub(filename=FILENAME):
    song = AudioSegment.from_file(filename)
    play(song)


if __name__ == '__main__':
    round = 0
    while True:
        print("====== round %d ======", round)
        AudioPlay_pydub()
        round += 1