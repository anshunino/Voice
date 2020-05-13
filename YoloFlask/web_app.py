# -*- coding: utf-8 -*-
"""
Created on Wed May 13 08:56:18 2020

@author: Anshuman
"""

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about/')
def about():
    return render_template("About.html")

@app.route('/rec')
def rec():
    return render_template("rec.html")

@app.route('/yolo')
def yolo():
    return render_template("yolo.html")

if __name__ == "__main__":
    app.run()
