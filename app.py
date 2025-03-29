import pandas as pd
import streamlit as st
from PIL import Image
import plotly.graph_objs as go
import plotly.express as px
from constant import *
from utils import *
#plt.style.use('dark_background')
st.set_page_config(page_title="EasyAD")

df = pd.read_csv('data/accuracy_table/{}.csv'.format('VUS-PR'))
df['filename'] = df['file']
df['dataset'] = df['file'].str.split('_').str[1]
df = df.set_index('filename')

with st.sidebar:
    st.markdown('# EasyAD Engine') 
    st.markdown('## Please Select :point_down:') 
    metric_name = st.selectbox('Evaluation Metric', list_measures)
    # metric_name = metric_name.split('-')[0]+'_'+metric_name.split('-')[1]
    
    container_dataset = st.container()  
    all_dataset = st.checkbox("Select all", key='all_dataset')
    if all_dataset: datasets = container_dataset.multiselect('Select Datasets', list(set(df['dataset'].values)), list(set(df['dataset'].values)))
    else: datasets = container_dataset.multiselect('Select Datasets', list(set(df['dataset'].values))) 
    
    container_category = st.container()
    all_catagory = st.checkbox("Select all", key='all_catagory')
    if all_catagory: methods_family = container_category.multiselect('Select EasyAD Category', list(method_group.keys()), list(method_group.keys()), key='selector_catagory_all')
    else: methods_family = container_category.multiselect('Select EasyAD Category', list(method_group.keys()), key='selector_catagory')

    container_method = st.container()
    best_method = st.checkbox("Best Variant", key='best_method')
    all_method = st.checkbox("All Variants", key='all_method')
    best_method_list = []
    all_method_list = []
    for method_g in methods_family:
        best_method_list.extend(best_variant[method_g])
        all_method_list.extend(method_group[method_g])    
    if best_method: methods_variant = container_method.multiselect('Select Automated Solutions', best_method_list, best_method_list, key='select_best_variant')
    elif all_method: methods_variant = container_method.multiselect('Select Automated Solutions', all_method_list, all_method_list, key='all_methods')
    else: methods_variant = container_method.multiselect('Select Automated Solutions', all_method_list, key='selector_methods')

df = pd.read_csv('data/accuracy_table/{}.csv'.format(metric_name))
df['filename'] = df['file']
df['dataset'] = df['file'].str.split('_').str[1]
df = df.set_index('filename')

tab_desc, tab_benchmark, tab_eva, tab_exploration = st.tabs(["Overview", "Benchmark", "Evaluation", "Data Exploration"]) 

with tab_desc:
    st.markdown("## :surfer: Dive into EasyAD")
    st.markdown("##### Automated Solutions for Time-Series Anomaly Detection")
    image = Image.open('figures/demo_sys.png')
    st.image(image)
    st.markdown(description_intro)
    st.markdown("#### User Manual")
    image = Image.open('figures/EasyAD_engine.png')
    st.image(image, caption='The main frames of EasyAD Engine')
    st.markdown(User_Manual)
    st.markdown(Contributors)


with tab_benchmark:
    st.markdown(benchmark_overview)
    image = Image.open('figures/framework.png')
    st.image(image, caption='Overview of TSB-AutoAD benchmark')
    st.markdown('#### 2. Taxonomy of Automated Solutions for TSAD')
    image = Image.open('figures/taxonomy_chronicle.png')
    st.image(image)

    # # labels = [
    # #     "EasyAD", 
    # #     "Model Selection", "Model Generation",
    # #     "Internal Evaluation", "Pretraining Based", "Ensembling Based", "Pseudo-label Based",
    # #     "EM&MV (2016)", "CQ (2016)", "MC (2022)", "Synthetic (2022)", "RA (2022)", "CLF (2020)", "RG (2008)", 
    # #     "UReg (2023)", "CFact (2023)", "kNN (2013)", "MetaOD (2021)", "ISAC (2010)",
    # #     "OE (2015)", "UE (2014)", "HITS (2023)", "Aug (2022)", "Clean (2023)", "Booster (2023)"
    # # ]
    # labels = [
    #     "EasyAD", 
    #     "Model Selection", "Model Generation",
    #     "Internal Evaluation", "Pretraining Based", "Ensembling Based", "Pseudo-label Based",
    #     "EM&MV", "CQ", "MC", "Synthetic", "RA", "CLF", "RG", "UReg", "CFact", "kNN", "MetaOD", "ISAC",
    #     "OE", "UE", "HITS", "Aug", "Clean", "Booster"
    # ]    
    # parents = [
    #     "", 
    #     "EasyAD", "EasyAD",
    #     "Model Selection", "Model Selection", "Model Generation", "Model Generation",
    #     "Internal Evaluation", "Internal Evaluation", "Internal Evaluation", "Internal Evaluation", "Internal Evaluation",
    #     "Pretraining Based", "Pretraining Based", "Pretraining Based", "Pretraining Based", "Pretraining Based", "Pretraining Based", "Pretraining Based",
    #     "Ensembling Based",  "Ensembling Based",  "Ensembling Based", "Pseudo-label Based", "Pseudo-label Based", "Pseudo-label Based"
    # ]

    # fig = go.Figure(go.Treemap(
    #     labels=labels,
    #     parents=parents,
    #     textfont=dict(size=18)
    # ))

    # fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    #     # treemapcolorway=["#3cb371", "#ffd700", "#3cb371", "#ffd700"])
    # st.plotly_chart(fig)

    # st.markdown(description_dataset)
    st.markdown(description_candidate)



with tab_eva:
    st.markdown('# Evaluation Overview')
    tab_acc, tab_time, tab_ms = st.tabs(["Accuracy Evaluation", "Runtime Analysis", "Model Selected Distribution"])
    with tab_acc:
        st.markdown('#### Evaluation Metrics: {}'.format(metric_name))
        # eva_type_col = st.columns([1])
        # with eva_type_col:
        #     eva_type = st.selectbox('Anomaly Type', ['All', 'Single Anomaly', 'Multiple Anomaly', 'Point Anomaly', 'Sequence Anomaly'])
            # container_eva_type = st.container()
            # eva_type = container_eva_type.multiselect('Anomaly Type', ['All', 'Single Anomaly', 'Multiple Anomaly', 'Point Anomaly', 'Sequence Anomaly'], key='selector_anomaly_type')

        eva_type_col, _, _ = st.columns([1, 1, 1])
        with eva_type_col:
            all_anomaly_type = st.checkbox("All Types of Anomaly", key='all_anomaly', value=True)
            container_eva_type = st.container()
            if all_anomaly_type: eva_type = container_eva_type.multiselect('Anomaly Type', ['All'], ['All'], key='all_selector_anomaly_type')
            else: eva_type = container_eva_type.multiselect('Anomaly Type', ['Single Anomaly', 'Multiple Anomaly', 'Point Anomaly', 'Sequence Anomaly'], key='selector_anomaly_type')

        Comparaed_Solution_Pool = list(baseline + methods_variant)
        
        if 'All' in eva_type:
            df_toplot = df.loc[df['dataset'].isin(datasets)][Comparaed_Solution_Pool]
        else:
            df_toplot = pd.DataFrame()
            df_filtered = df.loc[df['dataset'].isin(datasets)][Comparaed_Solution_Pool+['num_anomaly', 'point_anomaly', 'seq_anomaly']]
            if 'Single Anomaly' in eva_type:
                df_toplot = pd.concat([df_toplot, df_filtered[df_filtered['num_anomaly'] == 1]], ignore_index=True)
            if 'Multiple Anomaly' in eva_type:
                df_toplot = pd.concat([df_toplot, df_filtered[df_filtered['num_anomaly'] > 1]], ignore_index=True)
            if 'Point Anomaly' in eva_type:
                df_toplot = pd.concat([df_toplot, df_filtered[df_filtered['point_anomaly'] == 1]], ignore_index=True)
            if 'Sequence Anomaly' in eva_type:
                df_toplot = pd.concat([df_toplot, df_filtered[df_filtered['seq_anomaly'] == 1]], ignore_index=True)
            df_toplot = df_toplot.drop_duplicates()

        if len(datasets) == 0:
            st.markdown("#### :heavy_exclamation_mark: Note: Please select datasets in the left :point_left: panel")
        else:
            if len(df_toplot.columns) <= 5:        
                st.markdown("#### :heavy_exclamation_mark: Note: Please select automated solutions in the left :point_left: panel AND types of anomaly in the above :point_up_2:")

            else:
                df_toplot = df_toplot[Comparaed_Solution_Pool]
                plot_box_plot(target_df=df_toplot, methods_variant=Comparaed_Solution_Pool, metric_name=metric_name)
                st.markdown('<hr style="border:2px solid gray">', unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'>Table 1: Performance Comparision</h3>", unsafe_allow_html=True)
                st.dataframe(df_toplot)

    with tab_time:
        st.markdown(runtime_intro)
        det_time_df = pd.read_csv('data/runtime_table/detection_time.csv')
        exe_time_df = pd.read_csv('data/runtime_table/execution_time.csv')
        det_time, det_time_text, exe_time, exe_time_text, methods_det, methods_exe, acc_det, acc_exe = get_bubble_data(acc_df=df, det_time_df=det_time_df, exe_time_df=exe_time_df, methods_variant=Comparaed_Solution_Pool)

        if len(Comparaed_Solution_Pool) <= 5:        
            st.markdown("#### :heavy_exclamation_mark: Note: Please select automated solutions in the left :point_left: panel")
        else:    
            fig = go.Figure(data=[go.Scatter(
                x=methods_det,
                y=acc_det,
                text=det_time_text,
                mode='markers',
                marker=dict(
                    opacity=.7,
                    color='lightblue',
                    line = dict(width=1, color = '#1f77b4'),
                    size=det_time,
                    sizemode='area',
                    sizeref=2.*max(det_time)/(60.**2),
                    sizemin=4))])
            fig.update_xaxes(tickfont_size=14)
            fig.update_yaxes(tickfont_size=14)
            fig.update_layout(title=f'{metric_name} vs. Detection Time (Fast -> Slow)',
                            yaxis_title=f'{metric_name}',
                            showlegend=False, height=350, template="plotly_white", font=dict(size=19, color="black"))
            st.plotly_chart(fig)

            fig = go.Figure(data=[go.Scatter(
                x=methods_exe,
                y=acc_exe,
                text=exe_time_text,
                mode='markers',
                marker=dict(
                    opacity=.7,
                    color='lightblue',
                    line = dict(width=1, color = '#1f77b4'),
                    size=det_time,
                    sizemode='area',
                    sizeref=2.*max(det_time)/(60.**2),
                    sizemin=4))])
            fig.update_xaxes(tickfont_size=14)
            fig.update_yaxes(tickfont_size=14)
            fig.update_layout(title=f'{metric_name} vs. Execution Time (Fast -> Slow)',
                            yaxis_title=f'{metric_name}',
                            showlegend=False, height=350, template="plotly_white", font=dict(size=19, color="black"))
            st.plotly_chart(fig)


    with tab_ms:
        st.markdown('#### (1) Meta-learning-based Model Selection (ID vs. OOD)')
        # st.markdown('##### In-distribution vs. Out-of-distriution')
        st.markdown('* In-distribution (ID): The model selector is trained on all datasets.')
        st.markdown('* Out-of-distriution (OOD): The model selector is trained on all but one datasets.')

        df_AUC = pd.read_csv('data/accuracy_table/{}.csv'.format('AUC-PR'))
        df_AUC['filename'] = df_AUC['file']
        df_AUC['dataset'] = df_AUC['file'].str.split('_').str[1]
        df_AUC = df_AUC.set_index('filename')

        ms_distribution_pretraining_col, _, _ = st.columns([1, 1, 1])
        with ms_distribution_pretraining_col:
            container_ms_distribution_pretraining = st.container()
            ms_distribution_pretraining = container_ms_distribution_pretraining.multiselect('Pretrained Model Selector', methods_meta_id, key='ms_distribution_pretraining')
        if len(ms_distribution_pretraining) > 0:
            for method in ms_distribution_pretraining:
                st.markdown("##### :point_down: Overview of model selected distribution. `Count` indicates the frequency of each anomaly detector being chosen.")
                for case in [' (ID)', ' (OOD)']:
                    method_case = str(method)+case
                    globals()[method_case+'_dict'] = {det: 0 for det in Candidate_Model_Set}
                    for index, row in df_AUC.iterrows():
                        matching_columns = [col_name for col_name, col_value in row.items() if col_value == row[method_case]]
                        matching_elements = [element for element in matching_columns if element in Candidate_Model_Set]
                        globals()[method_case+'_dict'][matching_elements[0]] += 1
                    top_k = sorted(globals()[method_case+'_dict'].items(), key=lambda item: item[1], reverse=True)
                    keys, values = zip(*top_k)
                    keys_name = []
                    for i in keys:
                        keys_name.append(i)
                    fig = px.bar(x=keys_name, y=list(values), labels={'x':'Anomaly Detector', 'y':'Count'},title=method_case)
                    fig.update_layout(height=300)
                    st.plotly_chart(fig)

                st.markdown('<hr style="border:2px solid gray">', unsafe_allow_html=True)
                # st.markdown(f"##### :point_down: Pairwise Comparison of {method} (ID) and {method} (OOD) in terms of {metric_name}.")

                fig_pair = plt.figure(figsize=(6, 6))
                x = df[str(method)+' (ID)']
                y = df[str(method)+' (OOD)']

                # Draw diagonal line
                line_start = 0
                line_end = 1
                # Shade area above the diagonal line
                plt.fill_between([line_start, line_end], [line_start, line_end], [line_end, line_end], color='green', alpha=0.3)

                # Create scatter plot
                # plt.scatter(x, y, alpha=0.5, s=10)
                plt.scatter(x, y, s=10)
                plt.plot([line_start, line_end], [line_start, line_end], '-', color='orange', linewidth=2)

                # Label the axes
                plt.xlabel('ID', fontsize=12)
                plt.ylabel('OOD', fontsize=12)
                plt.title(f'Pairwise Comparison of {method} in {metric_name}', fontsize=14)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)

                # Set the limits of the axes
                plt.xlim(0, 1)
                plt.ylim(0, 1)

                fig2, ax2 = plt.subplots()
                ax2.plot(np.random.randn(50))

                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(fig_pair)

                with col2:
                    fig_avg = px.bar(x=['ID', 'OOD'], y=[df[str(method)+' (ID)'].mean(), df[str(method)+' (OOD)'].mean()], 
                                color=['ID', 'OOD'],
                                color_discrete_map={"ID": "blue", "OOD": "lightblue"})
                    fig_avg.update_layout(margin=dict(b=0), height=300, width=250, xaxis_title="", yaxis_title=metric_name, showlegend=False,     
                    title={
                        'text': f'Average {metric_name} of {method}',         
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'})
                    st.plotly_chart(fig_avg)


        st.markdown('#### (2) Internal Evaluation')

        ms_distribution_internal_col, _, _ = st.columns([1, 1, 1])
        with ms_distribution_internal_col:
            container_ms_distribution_internal = st.container()
            ms_distribution_internal = container_ms_distribution_internal.multiselect('Internal Evaluation', methods_ie, key='ms_distribution_internal')
        if len(ms_distribution_internal) > 0:
            st.markdown("##### :point_down: Overview of model selected distribution. `Count` indicates the frequency of each anomaly detector being chosen.")
            for method in ms_distribution_internal:
                globals()[method+'_dict'] = {det: 0 for det in Candidate_Model_Set}
                for index, row in df_AUC.iterrows():
                    matching_columns = [col_name for col_name, col_value in row.items() if col_value == row[method]]
                    matching_elements = [element for element in matching_columns if element in Candidate_Model_Set]
                    globals()[method+'_dict'][matching_elements[0]] += 1
                top_k = sorted(globals()[method+'_dict'].items(), key=lambda item: item[1], reverse=True)
                keys, values = zip(*top_k)
                keys_name = []
                for i in keys:
                    keys_name.append(i)
                fig = px.bar(x=keys_name, y=list(values), labels={'x':'Anomaly Detector', 'y':'Count'},title=method)
                st.plotly_chart(fig)

with tab_exploration:

    col_dataset_exp, col_ts_exp, col_meth_exp, col_gen_result = st.columns([1, 1, 1, 1])
    with col_dataset_exp:
        dataset_exp = st.selectbox('Pick a dataset', list(set(df['dataset'].values))+['Upload your own'])
    with col_ts_exp:
        time_series_selected_exp = st.selectbox('Pick a time series', list(df.loc[df['dataset']==dataset_exp].index))
    with col_meth_exp:
        if dataset_exp == 'Upload your own':
            method_selected_exp = st.selectbox('Pick a method', available_solution)
        else:
            method_selected_exp = st.selectbox('Pick a method', all_solution)
    with col_gen_result:
        gen_result = st.checkbox("Generate Result", key='gen_result')

    if method_selected_exp in methods_ie:
        st.markdown("You are using the Internal Evaluation Model Selection Method: {}".format(method_selected_exp))
    elif method_selected_exp in methods_meta: 
        st.markdown("You are using the Meta-learning-based Model Selection Method: {}".format(method_selected_exp))
    elif method_selected_exp in methods_ens: 
        st.markdown("You are using the Model Ensembling Method: {}".format(method_selected_exp))
    elif method_selected_exp in methods_generation: 
        st.markdown("You are using the Model Generation Method: {}".format(method_selected_exp))

    if dataset_exp == 'Upload your own':
        uploaded_ts = st.file_uploader("Upload your time series")
        if uploaded_ts is not None: 
            ts_data_raw = pd.read_csv(uploaded_ts).dropna()
    else:
        # path_ts = 'data/benchmark_ts/' + dataset_exp + '/' + time_series_selected_exp[:-4] + '.zip'
        # ts_data_raw = pd.read_csv(path_ts, compression='zip', header=None).dropna().to_numpy()
        path_ts = 'data/preloaded_ts/' + time_series_selected_exp
        ts_data_raw = pd.read_csv(path_ts).dropna()

    if dataset_exp != 'Upload your own' or uploaded_ts is not None:
        ts_data = ts_data_raw.iloc[:, 0:-1].values.astype(float).squeeze()
        label_data = ts_data_raw['Label'].astype(int).to_numpy()
            
        anom = add_rect(label_data, ts_data)
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=list(range(len(ts_data))), y=ts_data,
            name = "Time series", mode = 'lines',
            line = dict(color = 'blue', width=3), opacity = 1))
        fig.add_trace(go.Scattergl(
            x=list(range(len(ts_data))), y=anom,
            name = "Anomalies",
            mode = 'lines', line = dict(color = 'red', width=3), opacity = 1))
        if dataset_exp == 'Upload your own':
            fig.update_layout(title='User uploaded time series')
        else:
            fig.update_layout(title='File name: '+time_series_selected_exp)
        fig.update_layout(margin=dict(b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

        if gen_result:
            if not dataset_exp == 'Upload your own':
                st.markdown(f"* {metric_name} of {method_selected_exp} = {df.loc[time_series_selected_exp][method_selected_exp]}")
                method_display = set(Comparaed_Solution_Pool)
                method_display.add(method_selected_exp)
                df_display = df.loc[time_series_selected_exp, list(method_display)]
                sorted_df_display = df_display.to_frame().sort_values(by=time_series_selected_exp, ascending=False)
                st.markdown(f'* Ranking of {method_selected_exp} is: {sorted_df_display.index.tolist().index(method_selected_exp)}')

                st.markdown("* Ranked Evaluation details (Best -> Worst):")
                st.dataframe(sorted_df_display)

            else:
                pred_detector, success, vote_summary = run_model(ts_data.reshape(-1, 1), method_selected_exp)
                print('pred_detector: ', pred_detector)

                if success:
                    st.markdown(f"###### Voting details:")
                    st.bar_chart(vote_summary, height=150)
                    st.markdown(f"##### :star: Selected Model: {pred_detector}")

                    Anomaly_score = gen_as_from_det(ts_data.reshape(-1, 1), pred_detector)
                    if Anomaly_score is not None:

                        mean_score = np.mean(Anomaly_score)
                        std_dev = np.std(Anomaly_score)
                        threshold = mean_score + 3 * std_dev
                        pred_label = Anomaly_score > threshold
                        pred_anom = add_rect(pred_label, Anomaly_score)
                        # exceed_indices = np.where(Anomaly_score > threshold)[0]

                        fig = go.Figure()
                        fig.add_trace(go.Scattergl(
                            x=list(range(len(Anomaly_score))), y=Anomaly_score,
                            name = "Anomaly Score", mode = 'lines',
                            line = dict(color = 'lightblue', width=3), opacity = 1))

                        fig.add_hline(y=threshold, line=dict(color="red", width=2, dash="dash"), name="Threshold")

                        fig.add_trace(go.Scattergl(
                            x=list(range(len(Anomaly_score))), y=pred_anom,
                            name = "Predicted Anomalies",
                            mode = 'lines', line = dict(color = 'red', width=3), opacity = 1))

                        # for idx in exceed_indices:
                        #     fig.add_vrect(x0=idx-0.5, x1=idx+0.5, fillcolor="red", opacity=0.5, line_width=0)
                        fig.update_layout(title='Anomaly Score', height=300)

                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown(f"#### Failed at generating results... Please visit our Github repo for more information (https://github.com/TheDatumOrg/TSB-AutoAD)")