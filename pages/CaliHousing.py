import streamlit as st
import pydeck as pdk
import source.End_to_End_Project.CaliHousing.Decision_Tree_Regression as bai1
import source.End_to_End_Project.CaliHousing.Linear_Regression_UseModel as bai2
import source.End_to_End_Project.CaliHousing.Linear_Regression as bai3
import source.End_to_End_Project.CaliHousing.Random_Forest_Regression_Grid_Search_CV as bai4
import source.End_to_End_Project.CaliHousing.Random_Forest_Regression_Random_Search_CV_UseModel as bai5
import source.End_to_End_Project.CaliHousing.Random_Forest_Regression_Random_Search_CV as bai6
import source.End_to_End_Project.CaliHousing.Random_Forest_Regression as bai7
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url('https://image.winudf.com/v2/image/Y29tLnNpbXBsZWRyb2lkLndhbGxwYXBlcmdyYWRpZW50YmFja2dyb3VuZF9zY3JlZW5fMF8xNTI2OTY5MDEyXzAwMA/screen-0.jpg?fakeurl=1&type=.webp');
background-size: 80%;
background-position: right;
background-repeat: initial;
background-attachment: fixed;
background-repeat: no-repeat;
}"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("CaliHousing")
check = st.checkbox("Bảng dữ liệu")
if (check):
    st.dataframe(bai1.housing)

menu = st.sidebar.selectbox("menu", ["Decision_Tree_Regression", "Linear_Regression_UseModel", "Linear_Regression", "Random_Forest_Regression_Grid_Search_CV", "Random_Forest_Regression_Random_Search_CV_UseModel", "Random_Forest_Regression_Random_Search_CV", "Random_Forest_Regression"])

if menu == "Decision_Tree_Regression":
    st.title("Decision_Tree_Regression")
    st.line_chart(bai1.housing_labels)
    st.write("Sai số bình phương trung bình - train: ", '%.2f' % bai1.rmse_train)
    st.write("Sai số bình phương trung bình - cross-valid: ")
    st.write("Sai số bình phương trung bình - test: ", '%.2f' % bai1.rmse_test)

if menu == "Linear_Regression_UseModel":
    st.title("Linear_Regression_UseModel")
    st.subheader("Predictions")
    st.line_chart(bai2.pre)
    st.write("Sai số bình phương trung bình - train: ", '%.2f' % bai2.saiSo_train)
    st.write("Sai số bình phương trung bình - test: ", '%.2f' % bai2.saiSo_test)
    st.line_chart(bai2.y_predictions)
if menu== "Linear_Regression":
    st.title("Linear_Regression")
    st.subheader("Predictions")
    st.line_chart(bai3.pre1)
    st.write("Sai số bình phương trung bình - train: ", '%.2f' % bai3.saiSo_train)
    st.write("Sai số bình phương trung bình - test:", '%.2f' % bai3.saiSo_test)
if menu == "Random_Forest_Regression_Grid_Search_CV":
    st.title("Random_Forest_Regression_Grid_Search_CV")
    st.write("Sai số bình phương trung binh - train:", '%.2f' % bai4.rmse_train)
    st.write("Sai số bình phương trung bình - test:", '%.2f' % bai4.rmse_test)
if menu == "Random_Forest_Regression_Random_Search_CV_UseModel":
    st.title("Random_Forest_Regression_Random_Search_CV_UseModel")
    st.write("Sai số bình phương trung binh - train:", '%.2f' % bai5.rmse_train)
    st.write("Sai số bình phương trung bình - test:", '%.2f' % bai5.rmse_test)
if menu == "Random_Forest_Regression_Random_Search_CV":
    st.title("Random_Forest_Regression_Random_Search_CV")
    st.write("Sai số bình phương trung binh - train:", '%.2f' % bai6.rmse_train)
    st.write("Sai số bình phương trung bình - test:", '%.2f' % bai6.rmse_test) 
if menu == "Random_Forest_Regression":
    st.title("Random_Forest_Regression")
    st.write("Sai số bình phương trung binh - train:", '%.2f' % bai7.rmse_train)
    st.write("Sai số bình phương trung bình - test:", '%.2f' % bai7.rmse_test) 
