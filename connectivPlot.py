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
var = pd.read_csv("D:/PycharmProjects/StaticNetFormation/AllConnect-P=0.5-Q-0.5-C-0.005.csv")

x = var.Theta
y1 = var.N5
y2 = var.N10
y3 = var.N15
y4 = var.N20
y5 = var.N25
# Plot a horizontal line

plt.xlabel("Theta ("r"$\beta$)", fontdict = font1)
plt.ylabel("Network Connectivity(%)", fontdict = font1)
plt.plot(x, y1, '--C0', linewidth=2, label='N=5',  marker='o')
plt.plot(x, y2, '--C1', linewidth=2, label='N=10', marker='o')
plt.plot(x, y3, '--C2', linewidth=2, label='N=15', marker='o')
plt.plot(x, y4, '--C3', linewidth=2, label='N=20', marker='o')
plt.plot(x, y5, '--C4', linewidth=2, label='N=25', marker='o')
plt.legend()
plt.savefig('AllConnect-P=0.5-Q-0.5-C-0.005.png')
plt.savefig("AllConnect-P=0.5-Q-0.5-C-0.005.svg")
plt.savefig("AllConnect-P=0.5-Q-0.5-C-0.005.eps")
plt.show()