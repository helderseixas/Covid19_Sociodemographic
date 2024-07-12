import numpy as np
import statsmodels.api as sm
import pickle
import scipy.stats as stats
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

colors_default = {'extremamente_maior': '#80191C', 'muito_maior': '#D7191C', 'maior': '#FDAE61', 'medio': '#F09CFA',
                  'menor': '#ABD9E9', 'muito_menor': '#2c7bb6'}

def tunning_negative_binomial_model(x, y, list_offset):
    list_nb_glm_models = np.array([])
    list_llf_models = np.array([])
    list_alpha = np.arange(0.01, 2.1, 0.01)

    for alpha in list_alpha:
        nb_glm_model = sm.GLM(y, x, family=sm.families.NegativeBinomial(alpha=alpha), offset=list_offset).fit()
        llf = nb_glm_model.llf
        list_nb_glm_models = np.append(list_nb_glm_models, nb_glm_model)
        list_llf_models = np.append(list_llf_models, llf)

    # print('Selected alpha:', list_alpha[abs(list_llf_models) == min(abs(list_llf_models))][0])
    # return list_nb_glm_models[abs(list_llf_models) == min(abs(list_llf_models))][0]
    print('Selected alpha:', list_alpha[list_llf_models == max(list_llf_models)][0])
    return list_nb_glm_models[list_llf_models == max(list_llf_models)][0]

def save_model(model, name, directory_base='models'):
    with open(directory_base+'/'+name+'.pkl', 'wb') as file:
        pickle.dump(model, file)

def summarize_results(model):
    print(model.summary())
    print('llf:', round(model.llf, 2))
    try:
        print('deviance: ', round(model.deviance, 2))
        print('pearson_chi2: ', round(model.pearson_chi2, 2))
        print("Estimated dispersion: ", round(model.pearson_chi2/model.df_resid,2))
    except AttributeError:
        pass
    print('aic: ', round(model.aic, 2))


def outlier_analysis(model):
    influence = model.get_influence()
    standardized_residuals = model.resid_pearson

    cooks_d = influence.cooks_distance[0]

    plt.figure(figsize=(12, 6))

    # Plot standardized residuals
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(standardized_residuals)), standardized_residuals)
    plt.axhline(y=3, color='r', linestyle='--')
    plt.axhline(y=-3, color='r', linestyle='--')
    plt.title('Standardized Residuals')
    plt.xlabel('Observation')
    plt.ylabel('Standardized Residual')

    # Plot Cook's distance
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(cooks_d)), cooks_d)
    plt.axhline(y=4 / model.nobs, color='r', linestyle='--')
    plt.title("Cook's Distance")
    plt.xlabel('Observation')
    plt.ylabel("Cook's Distance")

    plt.tight_layout()
    plt.show()

    outliers = np.where(np.abs(standardized_residuals) > 3)[0]
    influential_points = np.where(cooks_d > 4 / model.nobs)[0]

    print("Quantity outliers (Standardized Residuals > 3):", len(outliers))
    print("Quantity influential Points (Cook's Distance > 4/n):", len(influential_points))

    return outliers, influential_points, cooks_d, standardized_residuals

def calculate_95_ci(data):
    mean = data.mean()
    sem = stats.sem(data)
    margin_of_error = sem * stats.t.ppf((1 + 0.95) / 2., len(data) - 1)
    return mean, mean - margin_of_error, mean + margin_of_error

def get_default_colors_heatmap():
    colors = [[0, 'darkblue'],
              [0.45, '#F0F0F0'],
              [0.55, '#F0F0F0'],
              [1, 'darkred']]
    return LinearSegmentedColormap.from_list('', colors)

def centimeter_to_inch(centimeters):
    return centimeters * 1/2.54

def get_default_colors_seaborn(n):
    colors_custom = [colors_default['muito_menor'], colors_default['menor'], colors_default['medio'],
                     colors_default['maior'], colors_default['muito_maior'], colors_default['extremamente_maior']]
    pallet_custom = sns.color_palette(colors_custom[:n])
    return pallet_custom

def get_default_colors_maps(n):
    colors_custom = [colors_default['muito_menor'], colors_default['menor'], colors_default['medio'],
                     colors_default['maior'], colors_default['muito_maior'], colors_default['extremamente_maior']]
    pallet_custom = ListedColormap(colors_custom[:n], name='map')
    return pallet_custom