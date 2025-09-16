import os, json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="LLM + SBERT + NCDM", layout="wide")
st.title("LLM-Augmented Q-matrix • SBERT Optimization • Simple NCDM")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Bloom levels per question")
    p = "reports/questions_bloom.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Run pipeline to generate Bloom labels.")

with col2:
    st.subheader("Optimized Q-matrix")
    q = "reports/q_matrix_optimized.csv"
    if os.path.exists(q):
        qdf = pd.read_csv(q)
        st.dataframe(qdf, use_container_width=True, hide_index=True)
    else:
        st.info("Run pipeline to build Q-matrix.")

st.markdown("---")
st.subheader("Training metrics")
m = "reports/training_metrics.csv"
if os.path.exists(m):
    md = pd.read_csv(m)
    st.dataframe(md, use_container_width=True, hide_index=True)
    fig = plt.figure()
    plt.plot(md["epoch"], md["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    st.pyplot(fig)
else:
    st.info("No training metrics yet. Run pipeline.")

st.markdown("---")
st.subheader("Student Mastery Heatmap & Feedback")
smp = "reports/student_mastery.csv"
if os.path.exists(smp):
    sm = pd.read_csv(smp).set_index("student_id")
    st.dataframe(sm.reset_index(), use_container_width=True, hide_index=True)
    fig = plt.figure()
    plt.imshow(sm.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(sm.columns)), sm.columns, rotation=45, ha="right")
    plt.yticks(range(len(sm.index)), sm.index)
    plt.title("Student Mastery (0..1)")
    st.pyplot(fig)

    sid = st.text_input("Check feedback for Student ID", "S01")
    if sid in sm.index:
        low = sm.loc[sid][sm.loc[sid] < 0.5].sort_values()
        if len(low)==0:
            st.success("Great! No weak concepts detected (≥ 0.5).")
        else:
            st.warning("Weak concepts detected: " + ", ".join([f"{k} ({v:.2f})" for k,v in low.items()]))
            st.write("Study plan:")
            for k, v in low.items():
                st.write(f"- **{k}**: review notes, attempt 3 practice problems, watch one tutorial video.")
else:
    st.info("No mastery file yet. Run pipeline to create reports/student_mastery.csv")
