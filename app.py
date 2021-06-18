from flask import Flask, render_template
import numpy as np
import cv2
import time
from gtts import gTTS 
import os
import matplotlib.pyplot as plt
import voice_code as v

app = Flask(__name__)


v.run()


@app.route('/')
@app.route('/home')
def home():
    return render_template('landing.html')

@app.route('/return-files/')
def return_files_tut():
    #code = voice.run()
    
    #voice.run()
    try:
        return render_template('home.html')
    except Exception as e:
        return str(e)



if __name__ == '__main__':
    app.run(debug=True)