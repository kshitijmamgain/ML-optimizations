#!/bin/bash
# runs experiment and then shuts down the script

python main.py --algorithm lgb --optimization hyperopt
sudo shutdown -h now