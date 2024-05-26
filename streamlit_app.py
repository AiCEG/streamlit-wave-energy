import streamlit as st
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

@st.cache_data
def load_all_csv_from_folder(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    data_frames = {}
    for file in csv_files:
        df = pd.read_csv(os.path.join(folder_path, file))
        data_frames[file] = df
    return data_frames

def perform_pca_and_plot(df, file_name):
    total_power = df['Total_Power']
    min_power, max_power = total_power.min(), total_power.max()
    bin_width = (max_power - min_power) / 4
    bins = [min_power + bin_width * i for i in range(5)]
    labels = ['Low', 'Medium', 'High', 'Very High']
    total_power_labels = pd.cut(total_power, bins=bins, labels=labels, include_lowest=True)
    label_encoder = LabelEncoder()
    total_power_encoded = label_encoder.fit_transform(total_power_labels)

    num_coords = 49 if '49' in file_name else 100
    all_coords = []
    for i in range(1, num_coords + 1):
        x_col, y_col = f'X{i}', f'Y{i}'
        coords = df[[x_col, y_col]].values
        all_coords.extend(coords)

    pca_df = pd.DataFrame(all_coords, columns=['X', 'Y'])
    pca_df['Total_Power_Label'] = total_power_encoded.repeat(num_coords)

    features = pca_df.columns
    x = pca_df.loc[:, features].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

    principalDf['Total_Power_Label'] = pca_df['Total_Power_Label'].values

    fig, ax = plt.subplots()
    for label, color in zip(range(len(labels)), ['b', 'g', 'r', 'c']):
        indicesToKeep = principalDf['Total_Power_Label'] == label
        ax.scatter(principalDf.loc[indicesToKeep, 'PC1'],
                   principalDf.loc[indicesToKeep, 'PC2'],
                   c=color,
                   s=5,
                   label=labels[label])
    ax.legend()
    ax.grid()

    plt.title(f'2 Component PCA for {file_name}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    st.pyplot(fig)

    st.write(f"Explained Variance Ratio of the first two components for {file_name}: {pca.explained_variance_ratio_}")

folder_path = 'csv'
data_frames = load_all_csv_from_folder(folder_path)

st.title('Filterable Tables and PCA for WEF CSV Files')
st.write('This app reads data from all WEF CSV files in the "csv" folder and allows you to filter the data for each file and view PCA results.')

def create_filters(df, prefix):
    filters = {}
    columns = df.columns.tolist()

    for col in columns:
        unique_values = df[col].unique()
        if len(unique_values) < 10:
            filters[col] = st.sidebar.multiselect(f'Select {col}', unique_values, key=f"{prefix}_{col}")
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                filters[col] = st.sidebar.slider(f'Select {col} range', min_val, max_val, (min_val, max_val), key=f"{prefix}_{col}")
            else:
                filters[col] = st.sidebar.text_input(f'Search {col}', key=f"{prefix}_{col}")

    return filters

for file_name, df in data_frames.items():
    st.header(f'Data from {file_name}')

    filters = create_filters(df, prefix=file_name)

    filtered_data = df
    for col, filter_val in filters.items():
        if isinstance(filter_val, list) and filter_val:
            filtered_data = filtered_data[filtered_data[col].isin(filter_val)]
        elif isinstance(filter_val, tuple):
            filtered_data = filtered_data[filtered_data[col].between(*filter_val)]
        elif isinstance(filter_val, str) and filter_val:
            filtered_data = filtered_data[filtered_data[col].str.contains(filter_val, case=False, na=False)]

    st.dataframe(filtered_data)

    st.subheader(f'PCA Analysis for {file_name}')
    perform_pca_and_plot(filtered_data, file_name)
