import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("K-Nearest Neighbors")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["bai01", "bai02", "Bai03", "Bai03a", "bai04", "bai08"])
with tab1:
    import source.KNN.Bai01 as bai1
    st.set_option('deprecation.showPyplotGlobalUse', False)
with tab3:
    import source.KNN.Bai03 as bai3
    st.text("Độ chính xác trên tập validation: ")
    st.write(bai3.val)
    st.text("Độ chính xác trên tập test: ")
    st.write(bai3.test)
with tab5:
    import source.KNN.Bai04 as bai4
