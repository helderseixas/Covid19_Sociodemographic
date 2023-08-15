import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma
import epyestim
import epyestim.covid19 as covid19
plt.style.use('seaborn-white')
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns

colors_default = {'extremamente_maior': '#80191C', 'muito_maior': '#D7191C', 'maior': '#FDAE61', 'medio': '#F09CFA',
                  'menor': '#ABD9E9', 'muito_menor': '#2c7bb6'}

def get_default_colors_maps(n):
    colors_custom = [colors_default['muito_menor'], colors_default['menor'], colors_default['medio'],
                     colors_default['maior'], colors_default['muito_maior'], colors_default['extremamente_maior']]
    pallet_custom = ListedColormap(colors_custom[:n], name='map')
    return pallet_custom

def get_default_colors_divergence_seaborn(n, reverse=False):
    colors_custom = [colors_default['muito_menor'], colors_default['menor'], colors_default['medio'],
                     colors_default['maior'], colors_default['muito_maior'], colors_default['extremamente_maior']]

    if reverse == True:
        pallet_custom = sns.color_palette(list(reversed(colors_custom[:n])))
    else:
        pallet_custom = sns.color_palette(colors_custom[:n])
    return pallet_custom

def get_default_colors_categorical_seaborn(n=5):
    colors_custom = ['#FEC4DC', '#3F80B3', '#610099', '#CCA86C', '#91E693']
    pallet_custom = sns.color_palette(colors_custom[:n])
    return pallet_custom

def get_default_colors_temporal_series_highlighting_peaks_seaborn(n=5):
    colors_custom = ['#C7C7C7', '#FF2B2B', '#3331E6', '#FFB60F', '#BF00F5']
    pallet_custom = sns.color_palette(colors_custom[:n])
    return pallet_custom

def get_default_colors_heatmap():
    colors = [[0, 'darkblue'],
              [0.45, '#F0F0F0'],
              [0.55, '#F0F0F0'],
              [1, 'darkred']]
    return LinearSegmentedColormap.from_list('', colors)

def centimeter_to_inch(centimeters):
    return centimeters * 1/2.54

