import streamlit as st
import pydeck as pdk
import source.End_to_End_Project.CaliHousing.Decision_Tree_Regression as bai1
import source.End_to_End_Project.CaliHousing.Linear_Regression_UseModel as bai2
import source.End_to_End_Project.CaliHousing.Linear_Regression as bai3


st.title("CaliHousing")
check = st.checkbox("Bảng dữ liệu")
if (check):
    st.dataframe(bai1.housing)


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Decision_Tree_Regression", "Linear_Regression_UseModel", "Linear_Regression", "Random_Forest_Regression_Grid_Search_CV", "Random_Forest_Regression_Random_Search_CV_UseModel", "Random_Forest_Regression_Random_Search_CV", "Random_Forest_Regression"])
with tab1:
    st.line_chart(bai1.housing_labels)
    
with tab2:
    st.subheader("Predictions")
    st.line_chart(bai2.pre)
    st.write("Sai số bình phương trung bình - train: ", bai2.saiSo_train)
    st.write("Sai số bình phương trung bình - test: ", bai2.saiSo_test)
    st.line_chart(bai2.y_predictions)
with tab3:
    st.subheader("Predictions")
    st.line_chart(bai3.pre1)
    st.write("Sai số bình phương trung bình - train: ", bai3.saiSo_train)
    st.write("Sai số bình phương trung bình - test:", bai3.saiSo_test)
    
