#!/usr/bin/env python
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

os.system('jupyter notebook --notebook-dir= ' + '"' + dir_path + '"')
