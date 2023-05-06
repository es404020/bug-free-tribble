
import streamlit as st
import pandas as pd
import os 

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup,pull,compare_models,save_model


# from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
# import warnings

# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

with st.sidebar:
    st.title('RegML')
    choice = st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This application allow you  build an automated ML pipeline using streamlit,Pandas Profiling and PyCaret")
    

# if os.path.exists("file.csv"):
#     df = pd.read_csv("file.csv",index_col=None)
#     st.dataframe(df)

if choice == "Upload":
    st.title("Upload your Data for modelling")
    file = st.file_uploader("Upload Your Dataset here")
    if file:
        df = pd.read_csv(file,index_col=None)
        # df.to_csv(str(file.name),index=None)
        df.to_csv("file.csv",index=None)
        st.dataframe(df)
        

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis ")
    profile_report = df.profile_report()
    st_profile_report(profile_report)


if choice == "ML":
    st.title("Machine learning")
    target = st.selectbox("Pick your target variable",df.columns)

    setup(data=df,target=target,silent=True)
    setup_df=pull()
    st.info("This is the ML experiment settings")
    st.dataframe( setup_df)
    best_model= compare_models()
    compare_df =pull()
    st.info("This is the ML Model")
    st.dataframe(compare_df)
    best_model
    save_model(best_model,'best_model')

    

if choice == "Download":
    with open("best_model.pkl",'rb') as f:
        st.download_button("Download model",f,"trined_model.pkl")