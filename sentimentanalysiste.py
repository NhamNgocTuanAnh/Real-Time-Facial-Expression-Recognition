import csv
import tkinter as tk
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
#working-large-csv-files-python/
from sqlalchemy import create_engine
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
filename = 'data.csv'
with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    highs = []

chunksize = 10 ** 6

for df in  pd.read_csv('data.csv', chunksize=chunksize):
    df

#plot individual lines with custom colors and labels
plt.plot(df['angry'], label='angry', color='#b26e39')
plt.plot(df['disgust'], label='disgust', color='#f09c52')
plt.plot(df['scared'], label='scared', color='#f7cda9')
plt.plot(df['happy'], label='happy', color='#dddddd')
plt.plot(df['sad'], label='sad', color='#9ecbed')
plt.plot(df['surprised'], label='scared', color='#4597da')
plt.plot(df['neutral'], label='scared', color='#2e6a99')
#add legend
plt.legend(title='Group')

#add axes labels and a title
plt.ylabel('Sales', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.title('Emotion-Anlal', fontsize=16)

#display plot
plt.show()
