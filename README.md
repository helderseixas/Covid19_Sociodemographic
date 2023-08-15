# Sociodemographic Factors Associated with Covid-19 Mortality in Brazilian Municipalities Across Three Years: An Approach Supported by Clustering

Welcome to the repository for the code and data of the paper "Sociodemographic Factors Associated with Covid-19 Mortality in Brazilian Municipalities Across Three Years: An Approach Supported by Clustering."

**Citation:**
If you utilize this code and data for your research, please cite our work as follows:

Hélder Seixas Lima, Petrônio Cândido de Lima e Silva, Wagner Meira Jr. et al. "Sociodemographic Factors Associated with Covid-19 Mortality in Brazilian Municipalities: An Approach Supported by Clustering," 25 July 2022, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-1889058/v1]

## Reproducibility

To reproduce the results presented in our paper, please follow the steps outlined below:

1. **Download and Prepare Datasets:**

   - **Covid-19 Data:**
     - Download Covid-19 data from the Sistema de Informação sobre Mortalidade (https://opendatasus.saude.gov.br/fa_IR/dataset/sim) for the years 2020, 2021, and 2022.
     - Place the downloaded files in the directory `covid/data/input`.

   - **Sociodemographic Data:**
     - Extract the file `covid/data/df_municipal.7z`.
     - This file contains the municipality database with various sociodemographic attributes.

   - **Mortality Level Data:**
     - We have provided the mortality level data used in our paper in the file `mortality_levels/data/data.7z`.
     - You can choose to analyze the same data we used or reproduce this data by executing all notebooks in the `mortality_levels` folder.

   - **Correlation Data:**
     - Similarly, the correlation data we used in our analysis is provided in the file `correlation_sociodemographic_covid/data/data.7z`.
     - You can decide whether to analyze the provided data or regenerate it by executing the notebooks in the `correlation_sociodemographic_covid` folder.

2. **Paper Figures:**

   - You can directly obtain the figures from our paper without running the scripts by extracting the following files:
     - `covid/images/images.7z`
     - `correlation_sociodemographic_covid/images/images.7z`

3. **Generate Results:**

   - Execute the notebooks in the following folders to generate results:
     - `covid`
     - `mortality_levels`
     - `correlation_sociodemographic_covid`

Please feel free to reach out if you have any questions or need assistance. Happy exploring and reproducing the findings of our study!