from bidict import bidict

# list_measures = ['VUS_PR', 'VUS_ROC', 'AUC_ROC', 'AUC_PR']
list_measures = ['VUS-PR', 'VUS-ROC', 'AUC-ROC', 'AUC-PR']

baseline = ['Oracle', 'GB', 'SS', 'Random (D)', 'Random (TS)']

methods_ie = ['EM', 'MV', 'CQ (XB)', 'CQ (Silhouette)', 'CQ (Hubert)', 'CQ (STD)', 'CQ (R2)', 'CQ (CH)', 'CQ (I-Index)', 'CQ (DB)', 'CQ (SD)', 'CQ (Dunn)', 
            'MC (1N)', 'MC (3N)', 'MC (5N)', 'Synthetic (sim. cutoff)', 'Synthetic (orig. cutoff)', 'Synthetic (sim. speedup)', 'Synthetic (orig. speedup)', 
            'Synthetic (sim. contextual)', 'Synthetic (orig. contextual)', 'Synthetic (sim. spike)', 'Synthetic (orig. spike)', 'Synthetic (sim. scale)', 
            'Synthetic (orig. scale)', 'Synthetic (sim. noise)', 'Synthetic (orig. noise)', 'RA (Borda)', 'RA (Trimmed Borda)', 'RA (Partial Borda)', 
            'RA (Partial Trimmed Borda)', 'RA (Kemeny)', 'RA (Trimmed Kemeny)']
methods_pretrain = ['kNN (ID)', 'kNN (OOD)', 'ISAC (ID)', 'ISAC (OOD)', 'Metaod (ID)', 'Metaod (OOD)', 
                    'UG (ID)', 'UG (OOD)', 'CLF (ID)', 'CLF (OOD)', 'UReg (ID)', 'UReg (OOD)', 'CFact (ID)', 'CFact (OOD)']
methods_pretrain_id = ['kNN', 'ISAC', 'Metaod', 'UG', 'CLF', 'UReg', 'CFact']
methods_ens = ['OE (Avg)', 'OE (Max)', 'OE (AOM)', 'UE', 'HITS']
methods_pseudo = ['Aug (Orig)', 'Aug (Ens)', 'Aug (Majority Voting)', 'Clean (Majority)', 'Clean (Individual)', 'Clean (Ratio)', 'Clean (Avg)', 'Booster']

method_group = {
       'Internal Evaluation': methods_ie,
       'Pretraining-based': methods_pretrain,
       'Ensembling-based': methods_ens,
       'Pseudo-label-based': methods_pseudo}

best_variant = {
       'Internal Evaluation': ['EM', 'CQ (XB)', 'MC (5N)', 'Synthetic (sim. cutoff)', 'RA (Borda)'],
       'Pretraining-based': ['kNN (ID)', 'ISAC (ID)', 'Metaod (ID)', 'UG (ID)', 'CLF (ID)', 'UReg (ID)', 'CFact (ID)'],
       'Ensembling-based': ['OE (AOM)', 'UE', 'HITS'],
       'Pseudo-label-based': ['Aug (Ens)', 'Clean (Majority)', 'Booster']}

available_solution = ['UReg', 'CLF']

all_solution = ['EM', 'MV', 'CQ (XB)', 'CQ (Silhouette)', 'CQ (Hubert)', 'CQ (STD)', 'CQ (R2)', 'CQ (CH)', 'CQ (I-Index)', 'CQ (DB)', 'CQ (SD)', 'CQ (Dunn)', 
            'MC (1N)', 'MC (3N)', 'MC (5N)', 'Synthetic (sim. cutoff)', 'Synthetic (orig. cutoff)', 'Synthetic (sim. speedup)', 'Synthetic (orig. speedup)', 
            'Synthetic (sim. contextual)', 'Synthetic (orig. contextual)', 'Synthetic (sim. spike)', 'Synthetic (orig. spike)', 'Synthetic (sim. scale)', 
            'Synthetic (orig. scale)', 'Synthetic (sim. noise)', 'Synthetic (orig. noise)', 'RA (Borda)', 'RA (Trimmed Borda)', 'RA (Partial Borda)', 
            'RA (Partial Trimmed Borda)', 'RA (Kemeny)', 'RA (Trimmed Kemeny)', 'kNN (ID)', 'kNN (OOD)', 'ISAC (ID)', 'ISAC (OOD)', 'Metaod (ID)', 'Metaod (OOD)', 
            'UG (ID)', 'UG (OOD)', 'CLF (ID)', 'CLF (OOD)', 'UReg (ID)', 'UReg (OOD)', 'CFact (ID)', 'CFact (OOD)', 'OE (Avg)', 'OE (Max)', 'OE (AOM)', 'UE', 'HITS',
            'Aug (Orig)', 'Aug (Ens)', 'Aug (Majority Voting)', 'Clean (Majority)', 'Clean (Individual)', 'Clean (Ratio)', 'Clean (Avg)', 'Booster']

description_intro = f"""

Despite the recent focus on time-series anomaly detection, the effectiveness of the proposed anomaly detectors is restricted to specific domains. It is worth noting that a model that performs well on one dataset may not perform well on another. Therefore, how to select the optimal model for a particular dataset has emerged as a pressing issue. 
However, there is a noticeable gap in the existing literature when it comes to providing a comprehensive review of the ongoing efforts in this field. The evaluation of proposed methods across different datasets and under varying assumptions may create an illusion of progress. 
Motivated by the limitations above, we introduce the AutoTSAD Engine, a modular web interface designed to facilitate the exploration of the first comprehensive benchmark for automated time-series anomaly detection. AutoTSAD Engine enables rigorous statistical analysis of 18 automated solutions across 18 public datasets, incorporating a two-dimensional evaluation that includes both (i) accuracy and (ii) runtime analysis. And it allows users to assess the performance of various methods both (i) globally, by providing aggregated evaluation across datasets, and (ii) individually, offering fine-grained analysis per time series. Furthermore, the engine accommodates the processing of user-uploaded data, enabling users to experiment with different model selection strategies on their own datasets. 
Our goal for the interactive AutoTSAD Engine is to help users gain insights into different methods and to facilitate a more intuitive comprehension of the performance disparities among these methods.

Github repo: https://github.com/TheDatumOrg/AutoTSAD

#### Contributors

* [Qinghua Liu](https://qhliu26.github.io) (The Ohio State University)
* [Seunghak Lee](https://www.cs.cmu.edu/~seunghak) (Meta)
* [John Paparrizos](https://www.paparrizos.org) (The Ohio State University)

#### User Manual

"""

benchmark_overview = f"""
#### 1. Benchamrk Overview

We use M1, M2, and M3 to represent the candidate models. (a) depicts the standard evaluation pipeline for anomaly detectors. (b) depicts the pretraining pipeline for pretraining-based model selectors. (c) outlines the process for Model Selection which includes two main categories: (c.1) Internal Evaluation and (c.2) Pretraining-based Method. The output is the chosen anomaly detector that can then be applied to the time series data. (d) shows the approach for Model Generation which includes: (d.1) Ensembling-based and (d.2) Pseudo-label-based Methods. The result can be considered as an anomaly detector on its own.

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

A value of 1 in `Win` indicates using the max periodicity of the time series as the sliding window length, and 2 denotes the second-max periodicity. A value of 0 implies that we do not apply the sliding window strategy, with each time step processed individually. `Model Hyperparameter` outlines the different hyperparameter settings (see TSB for detailed definitions). We use a (Win, HP) tuple to specify hyperparameter configurations for each candidate model in `Candidate Model`.

| Method | Win        | Model Hyperparameter                                      | Candidate Model              |
|--------|------------|-----------------------------------------------------------|------------------------------|
| IForest| [0,1,2,3]  | n_estimators=[20, 50, 75, 100, 150, 200]                  | (3, 200), (1,100), (0,200)   |
| LOF    | [1,2,3]    | n_neighbors=[10, 30, 60]                                  | (3,60), (1,30)               |
| MP     | [1,2,3]    | cross_correlation=[False,True]                            | (2,False), (1,True)          |
| PCA    | [1,2,3]    | n_components=[0.25, 0.5, 0.75, None]                      | (3,None), (1,0.5)            |
| NORMA  | [1,2,3]    | clustering=[hierarchical, kshape]                         | (1,hierarchical), (3,shape)  |
| HBOS   | [1,2,3]    | n_bins=[5, 10, 20, 30, 40, 50]                            | (3,20), (1,40)               |
| POLY   | [1,2,3]    | power=[1, 2, 3, 4, 5, 6]                                  | (3,5), (2,1)                 |
| OCSVM  | [1,2,3]    | kernel_set=[linear, poly, rbf, sigmoid]                   | (1,rbf), (3,poly)            |
| AE     | [1,2,3]    | hidden_neuron=[[64, 32, 32, 64], [32, 16, 32]], norm=[bn, dropout] | (1,[32, 16, 32],bn), (2, [64, 32, 32, 64],dropout) |
| CNN    | [1,2,3]    | num_channel=[[32, 32, 40], [8, 16, 32, 64]] activation=[relu, sigmoid, tanh] | (2,[32, 32, 40],relu), (3,[8, 16, 32, 64],sigmoid) |
| LSTM   | [1,2,3]    | hidden_dim=[32, 64], activation=[relu, sigmoid]           | (1,64,relu), (3,64,sigmoid)  |
"""

runtime_intro = f"""

##### Reminder
* Detection Time: The duration required to obtain a detection result (i.e., the anomaly score).
* Execution Time: The time needed to identify the selected model from a given time series.

"""

Candidate_Model_Set = ['IForest_3_200', 'IForest_1_100', 'IForest_0_200', 'LOF_3_60', 'LOF_1_30', 
                       'MP_2_False', 'MP_1_True', 'PCA_3_None', 'PCA_1_0.5', 'NORMA_1_hierarchical', 'NORMA_3_kshape', 
                       'HBOS_3_20', 'HBOS_1_40', 'POLY_3_5', 'POLY_2_1', 'OCSVM_1_rbf', 'OCSVM_3_poly',
                       'AE_1_1_bn', 'AE_2_0_dropout', 'CNN_2_0_relu', 'CNN_3_1_sigmoid', 'LSTM_1_1_relu', 'LSTM_3_1_sigmoid']

det_name_mapping = bidict({'IForest_3_200':'IF1', 'IForest_1_100':'IF2', 'IForest_0_200':'IF3', 'LOF_3_60':'LOF1', 'LOF_1_30':'LOF2', 'MP_2_False':'MP1', 'MP_1_True':'MP2', 
                       'PCA_3_None':'PCA1', 'PCA_1_0.5':'PCA2', 'NORMA_1_hierarchical':'NORMA1', 'NORMA_3_kshape':'NORMA2', 'HBOS_3_20':'HBOS1', 'HBOS_1_40':'HBOS2', 
                       'POLY_3_5':'POLY1', 'POLY_2_1':'POLY2', 'OCSVM_1_rbf':'OCSVM1', 'OCSVM_3_poly':'OCSVM2', 'AE_1_1_bn':'AE1', 'AE_2_0_dropout':'AE2', 
                       'CNN_2_0_relu':'CNN1', 'CNN_3_1_sigmoid':'CNN2', 'LSTM_1_1_relu':'LSTM1', 'LSTM_3_1_sigmoid':'LSTM2'})