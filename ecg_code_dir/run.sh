#!/bin/bash

apt-get -y install python3-pip
pip3 install keras==2.1.6 
python3 /dbc/code/train_keras_2.py
