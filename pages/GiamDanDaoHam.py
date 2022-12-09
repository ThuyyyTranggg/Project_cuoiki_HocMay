import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

#import source.GiamDanDaoHam.Bai01 as bai1
import source.GiamDanDaoHam.Bai02 as bai2
import source.GiamDanDaoHam.Bai02a as bai2a
st.title("GIẢM DẦN - ĐẠO HÀM")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["bai01", "bai02", "Bai02a", "Bai03", "bai04", "bai05", "temp"])
with tab1:
    import source.GiamDanDaoHam.Bai01 as bai1
    st.set_option('deprecation.showPyplotGlobalUse', False)
with tab2:
    st.write("hi")
    st.line_chart(bai2.z)
    st.line_chart(bai2.t)
with tab3:
    st.line_chart(bai2a.z)