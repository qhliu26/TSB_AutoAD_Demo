list_measures = ['VUS-PR', 'VUS-ROC', 'AUC-PR', 'AUC-ROC', 'Standard-F1', 'Affiliation-F', 'Event-based-F1', 'PA-F1', 'R-based-F1']

baseline = ['Oracle', 'SS', 'Random', 'GB', 'FM']

methods_ie = ['Synthetic (Orig-contextual)', 'Synthetic (Orig-cutoff)', 'Synthetic (Orig-noise)', 'Synthetic (Orig-scale)', 'Synthetic (Orig-speedup)', 'Synthetic (Orig-spikes)', 'Synthetic (STL-contextual)', 'Synthetic (STL-cutoff)', 'Synthetic (STL-noise)', 'Synthetic (STL-scale)', 'Synthetic (STL-speedup)', 'Synthetic (STL-spikes)',
              'UEC (EM)', 'UEC (MV)', 'MC (3)', 'MC (5)', 'MC (7)', 'MC (9)', 'MC (12)',
       'CQ (CH)', 'CQ (DB)', 'CQ (Dunn)', 'CQ (Hubert)', 'CQ (I-Index)', 'CQ (R2)', 'CQ (SD)', 'CQ (Silhouette)', 'CQ (STD)', 'CQ (XBS)',
              'TSADAMS (Borda)', 'TSADAMS (Kemeny)', 'TSADAMS (Trimmed Kemeny)', 'TSADAMS (Partial Borda)', 'TSADAMS (Trimmed Borda)', 'TSADAMS (MIM)']
methods_meta = ['MSAD (ID)', 'MSAD (OOD)', 'SATzilla (ID)', 'SATzilla (OOD)',  'UReg (ID)', 'UReg (OOD)', 
    'CFact (ID)', 'CFact (OOD)', 'ISAC (ID)', 'ISAC (OOD)', 'ARGOSMART (ID)', 'ARGOSMART (OOD)', 'MetaOD (ID)', 'MetaOD (OOD)']
methods_meta_id = ['MSAD', 'SATzilla', 'UReg', 'CFact', 'ISAC', 'ARGOSMART', 'MetaOD']
methods_ens = ['OE (AOM)', 'OE (AVG)', 'OE (MAX)', 'HITS', 'IOE', 'SELECT (V)', 'SELECT (H)', 'AutoTSAD']
methods_generation = ['AutoOD-A (Ensemble)', 'AutoOD-A (Orig)', 'AutoOD-A (Majority Vote)', 'AutoOD-C (Majority)', 'AutoOD-C (Ratio)', 'AutoOD-C (Average)', 'AutoOD-C (Individual)',
              'UADB (Orig)', 'UADB (Base Mean)', 'UADB (Base STD)', 'UADB (Base Mean Cascade)', 'UADB (Base STD Cascade)']

method_group = {
       'Internal Evaluation': methods_ie,
       'Meta-learning-based': methods_meta,
       'Ensembling': methods_ens,
       'Generation': methods_generation}

best_variant = {
       'Internal Evaluation': ['CQ', 'UEC', 'MC', 'Synthetic', 'TSADAMS'],
       'Meta-learning-based': ['MSAD (ID)', 'SATzilla (ID)', 'UReg (ID)', 'CFact (ID)', 'ISAC (ID)', 'ARGOSMART (ID)', 'MetaOD (ID)'],
       'Ensembling': ['OE', 'HITS', 'IOE', 'SELECT', 'AutoTSAD'],
       'Generation': ['AutoOD-A', 'AutoOD-C', 'UADB']
}

available_solution = ['MSAD', 'SATzilla']

all_solution = methods_ie + methods_meta + methods_ens + methods_generation

description_intro = f"""

Despite the recent focus on time-series anomaly detection, the effectiveness of the proposed anomaly detectors is restricted to specific domains. A model that performs well on one dataset may not perform well on another. Therefore, how to develop automated solutions for anomaly detection for a particular dataset has emerged as a pressing issue. However, there is a noticeable gap in the literature regarding providing a comprehensive review of the ongoing efforts toward automated solutions for selecting or generating scores in an automated manner. Conducting a meta-analysis of proposed methods is challenging due to: (i) their evaluation across limited datasets; (ii) different assumptions on application scenarios; and (iii) the absence of evaluations for out-of-distribution performance. Motivated by the limitations above, we introduce the EasyAD, a modular web engine designed to facilitate the exploration of the first comprehensive benchmark for automated time-series anomaly detection. The EasyAD engine enables rigorous statistical analysis of 20 automated methods and 70 of their variants across the TSB-AD benchmark, a recently curated, heterogeneous dataset spanning nine application domains. The engine supports a two-dimensional evaluation framework, incorporating both accuracy and runtime performance. Our engine allows users to assess the performance of various methods per dataset and per instance, which offers fine-grained analysis per time series. Furthermore, the engine accommodates the processing of user-uploaded data, enabling users to experiment with different model selection strategies on their own datasets. Our goal for EasyAD is to help users gain insights into different methods and to facilitate a more intuitive comprehension of the performance disparities among these methods.

* Github repo: https://github.com/TheDatumOrg/TSB-AutoAD

"""

User_Manual = f"""

(a) Finding the overall best automated solutions

(b) Investigating the influence of different anomaly types

(c) Understanding accuracy to runtime trade-off

(d) Exploring the model selected distribution and the effect of domain shift

(e) Testing on your own data (Be sure to follow the format of TSB-AD Benchmark)

"""


Contributors = f"""

#### Contributors

* [Qinghua Liu](https://qhliu26.github.io) (The Ohio State University)
* [Seunghak Lee](https://www.cs.cmu.edu/~seunghak) (Meta)
* [John Paparrizos](https://www.paparrizos.org) (The Ohio State University)

"""

benchmark_overview = f"""
#### 1. Benchamrk Overview

An overview of automated solution pipeline. We use M1, M2, and Mn to represent the candidate models.

"""


description_dataset = f"""

#### 3. Dataset Overview

| Dataset      | Description                                  |
|--------------|----------------------------------------------|
| Dodgers      | unusual traffic after a Dodgers game         |
| ECG          | standard electrocardiogram dataset           |
| IOPS         | performance indicators of a machine          |
| KDD21        | UCR Anomaly Archive                          |
| MGAB         | Mackey-Glass time series                     |
| NAB          | web-related data                             |
| SensorScope  | environmental data                           |
| YAHOO        | Yahoo production systems data                |
| NASA-MSL     | Curiosity Rover telemetry data               |
| NASA-SMAP    | Soil Moisture Active Passive satellite data |
| Daphnet      | acceleration sensors                         |
| GHL          | gasoil heating loop telemetry                |
| Genesis      | portable pick-and-place demonstrator         |
| MITDB        | ambulatory ECG recordings                    |
| OPP          | motion sensors                               |
| Occupancy    | room occupancy data                          |
| SMD          | server machine telemetry                     |
| SVDB         | ECG recordings                               |
"""


description_candidate = f"""

#### 4. Candidate Model Set

##### (i) Statistical Method

| Algorithm    | Description|
|:--|:---------|
|(Sub)-MCD|is based on minimum covariance determinant, which seeks to find a subset of all the sequences to estimate the mean and covariance matrix of the subset with minimal determinant. Subsequently, Mahalanobis distance is utilized to calculate the distance from sub-sequences to the mean, which is regarded as the anomaly score.|
|(Sub)-OCSVM|fits the dataset to find the normal data's boundary by maximizing the margin between the origin and the normal samples.|
|(Sub)-LOF|calculates the anomaly score by comparing local density with that of its neighbors.|
|(Sub)-KNN|produces the anomaly score of the input instance as the distance to its $k$-th nearest neighbor.|
|KMeansAD|calculates the anomaly scores for each sub-sequence by measuring the distance to the centroid of its assigned cluster, as determined by the k-means algorithm.|
|CBLOF|is clluster-based LOF, which calculates the anomaly score by first assigning samples to clusters, and then using the distance among clusters as anomaly scores.|
|POLY|detect pointwise anomolies using polynomial approximation. A GARCH method is run on the difference between the approximation and the true value of the dataset to estimate the volatility of each point.|
|(Sub)-IForest|constructs the binary tree, wherein the path length from the root to a node serves as an indicator of anomaly likelihood; shorter paths suggest higher anomaly probability.|
|(Sub)-HBOS|constructs a histogram for the data and uses the inverse of the height of the bin as the anomaly score of the data point.|
|KShapeAD| identifies the normal pattern based on the k-Shape clustering algorithm and computes anomaly scores based on the distance between each sub-sequence and the normal pattern. KShapeAD improves KMeansAD as it relies on a more robust time-series clustering method and corresponds to an offline version of the streaming SAND method.|
|MatrixProfile|identifies anomalies by pinpointing the subsequence exhibiting the most substantial nearest neighbor distance.|
|(Sub)-PCA|projects data to a lower-dimensional hyperplane, with significant deviation from this plane indicating potential outliers.|
|RobustPCA|is built upon PCA and identifies anomalies by recovering the principal matrix.|
|EIF|is an extension of the traditional Isolation Forest algorithm, which removes the branching bias using hyperplanes with random slopes.|
|SR| begins by computing the Fourier Transform of the data, followed by the spectral residual of the log amplitude. The Inverse Fourier Transform then maps the sequence back to the time domain, creating a saliency map. The anomaly score is calculated as the relative difference between saliency map values and their moving averages.|
|COPOD|is a copula-based parameter-free detection algorithm, which first constructs an empirical copula, and then uses it to predict tail probabilities of each given data point to determine its level of extremeness.|
|Series2Graph| converts the time series into a directed graph representing the evolution of subsequences in time. The anomalies are detected using the weight and the degree of the nodes and edges respectively.|
|SAND| identifies the normal pattern based on clustering updated through arriving batches (i.e., subsequences) and calculates each point's effective distance to the normal pattern.|


##### (ii) Neural Network-based Method

| Algorithm    | Description|
|:--|:---------|
|AutoEncoder|projects data to the lower-dimensional latent space and then reconstruct it through the encoding-decoding phase, where anomalies are typically characterized by evident reconstruction deviations.|
|LSTMAD|utilizes Long Short-Term Memory (LSTM) networks to model the relationship between current and preceding time series data, detecting anomalies through discrepancies between predicted and actual values.|
|Donut|is a Variational AutoEncoder (VAE) based method and preprocesses the time series using the MCMC-based missing data imputation technique.|
|CNN|employ Convolutional Neural Network (CNN) to predict the next time stamp on the defined horizon and then compare the difference with the original value.|
|OmniAnomaly|is a stochastic recurrent neural network, which captures the normal patterns of time series by learning their robust representations with key techniques such as stochastic variable connection and planar normalizing flow, reconstructs input data by the representations, and use the reconstruction probabilities to determine anomalies.|
|USAD|is based on adversely trained autoencoders, and the anomaly score is the combination of discriminator and reconstruction loss.|
|AnomalyTransformer|utilizes the `Anomaly-Attention' mechanism to compute the association discrepancy.|
|TranAD|is a deep transformer network-based method, which leverages self-conditioning and adversarial training to amplify errors and gain training stability.|
|TimesNet|is a general time series analysis model with applications in forecasting, classification, and anomaly detection. It features TimesBlock, which can discover the multi-periodicity adaptively and extract the complex temporal variations from transformed 2D tensors by a parameter-efficient inception block.|
|FITS|is a lightweight model that operates on the principle that time series can be manipulated through interpolation in the complex frequency domain.|

##### (iii) Foundation Model-based Method

| Algorithm    | Description|
|:--|:---------|
|OFA|finetunes pre-trained GPT-2 model on time series data while keeping self-attention and feedforward layers of the residual blocks in the pre-trained language frozen.|
|Lag-Llama|is the first foundation model for univariate probabilistic time series forecasting based on a decoder-only transformer architecture that uses lags as covariates.|
|Chronos|tokenizes time series values using scaling and quantization into a fixed vocabulary and trains the T5 model on these tokenized time series via the cross-entropy loss.|
|TimesFM|is based on pretraining a decoder-style attention model with input patching, using a large time-series corpus comprising both real-world and synthetic datasets.|
|MOMENT|is pre-trained T5 encoder based on a masked time-series modeling approach.|


"""

runtime_intro = f"""

##### Reminder
* Detection Time: The duration required to obtain a detection result (i.e., the anomaly score).
* Execution Time: The time needed to identify the selected model from a given time series.

"""

Candidate_Model_Set = ['Sub_IForest', 'IForest', 'Sub_LOF', 'LOF', 'POLY', 'MatrixProfile', 'KShapeAD', 'SAND', 'Series2Graph', 'SR', 'Sub_PCA', 'Sub_HBOS', 'Sub_OCSVM', 
        'Sub_MCD', 'Sub_KNN', 'KMeansAD_U', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 
        'TimesNet', 'FITS', 'OFA', 'Lag_Llama', 'Chronos', 'TimesFM', 'MOMENT_ZS', 'MOMENT_FT']