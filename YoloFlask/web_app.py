# -*- coding: utf-8 -*-
"""
Created on Wed May 13 08:56:18 2020

@author: Anshuman
"""
import time
from flask import Flask, render_template
from yolo_camera import yolo_cam

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about/')
def about():
    return render_template("About.html")

@app.route('/rec')
def rec():    
    yolo_cam()
    return render_template("rec1.html")

@app.route('/yolo')
def yol():
    return render_template("yolo.html")

if __name__ == "__main__":
    app.run()
