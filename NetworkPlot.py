import os
import random
import cv2
import networkx
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import pandas
import pandas as pd
import openpyxl
from openpyxl import Workbook
import numpy as np

#######################################
font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}
###########################################

var = pd.read_excel("D:/PycharmProjects/StaticNetFormation/AllMaxP0.5-Q0.5-C-0.009.xlsx")

x = var.Theta
y1 = var.agents5
y2 = var.agents10
y3 = var.agents15
y4 = var.agents20
y5 = var.agents25
# Plot a horizontal line
#plt.title("Benefit Vs Density", fontdict = font1)
#plt.rcParams['font.size'] = '22'
#fig = plt.figure(figsize=(19.20,10.80))
#fig = plt.figure()
#fig.set_size_inches(18.5, 10.5, forward=True)
plt.xlabel("Benefit ("r"$\beta$)", fontdict = font1)
plt.ylabel("Density", fontdict = font1)
plt.plot(x, y1, '--r', linewidth=2, label='N=5', marker='o')
plt.plot(x, y2, '--b', linewidth=2, label='N=10', marker='o')
plt.plot(x, y3, '--g', linewidth=2, label='N=15', marker='o')
plt.plot(x, y4, '--c', linewidth=2, label='N=20', marker='o')
plt.plot(x, y5, '--m', linewidth=2, label='N=25', marker='o')
plt.legend()
plt.savefig('AllMax-P=0.9-Q=0.1-C=0.001.png')
plt.savefig("AllMax-P=0.9-Q=0.1-C=0.001.svg")
plt.savefig("AllMax-P=0.9-Q=0.1-C=0.001.eps")
plt.show()