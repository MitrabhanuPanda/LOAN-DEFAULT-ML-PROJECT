import streamlit as st
import pandas as pd
import numpy as np
import pickle

model=pickle.load(open("D:\\DV ANALYTIC\\TOPICS\\4. DATA MINING\\ML PROJECT\\1. BANKING\\Bank Domain\\MODEL\\pipe2.pkl","rb"))
df=pickle.load(open(r"MODEL/df.pkl","rb"))
df1=pd.read_csv(r"D:\DV ANALYTIC\TOPICS\4. DATA MINING\ML PROJECT\1. BANKING\Bank Domain\final_data.csv")


st.title("LOAN DEFAULTER MODEL PREDICTION")

age = st.number_input("ENTER AGE OF THE PERSON",value=None,step=1, max_value=70,min_value=21)
loan_percent = st.number_input("PERCENTAGE OF INCOME",value=None, max_value=1.00,min_value=0.00)
int_rat = st.number_input("INTEREST RATE",value=None, max_value=100.00,min_value=0.00)
ownership=st.selectbox("HOME OWNERSHIP",["RENT", "MORTGAGE", "OWN", "OTHER"])
per_income=st.number_input("PERSON INCOME",value=None,step=1)

if st.button("PREDICTION"):

    if ownership=="RENT":
        ownership=0
    elif ownership=="OWN":
        ownership=1
    elif ownership=="MORTGAGE":
        ownership=2
    else:
        ownership=3


    new_query=np.array([[age,loan_percent,int_rat,ownership,per_income]])
    new_query=new_query.reshape(1,5)

    prob_pred=model.predict_proba(new_query)
    
    pred_out=np.where(prob_pred[:, 1]>=0.7,1,0)

    if int(pred_out)==1:
        st.title(f"THE PREDICTION IS : DEFAULT" )
    else:
        st.title(f"THE PREDICTION IS : NON DEFAULT" )
   














