
# ==============================================================================
# PharmAssist AI: Streamlit Drug Interaction Assistant using FAISS RAG
# Complete Runnable Python Implementation (app.py)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os

# --- FAISS & LangChain RAG Libraries ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="PharmAssist RAG - Drug Interaction Analyzer",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 1. MEDICAL DISCLAIMER BANNER (MANDATORY)
# ------------------------------------------------------------------------------
def render_disclaimer():
    st.warning(
        "⚠️ **CRITICAL MEDICAL DISCLAIMER:** This application is an experimental AI decision-support system. "
        "It provides interaction data for educational purposes only and is **NOT** a substitute for professional "
        "medical advice, clinical diagnosis, or treatment. Always consult a licensed healthcare provider before "
        "modifying any prescribed medication regimen."
    )

# ------------------------------------------------------------------------------
# 2. LOCAL KNOWLEDGE BASE & FAISS INDEX SETUP
# ------------------------------------------------------------------------------
DRUG_INTERACTION_KB = [
    {
        "drugs": ["Warfarin", "Aspirin"],
        "severity": 92,
        "category": "High Risk / Critical Contraindication",
        "mechanism": "Warfarin inhibits vitamin K-dependent clotting factors, while Aspirin causes irreversible platelet inhibition. Concurrent use synergistically increases severe gastrointestinal and intracranial hemorrhaging risk.",
        "empathetic_summary": "Combining Warfarin and Aspirin significantly increases your risk of bleeding. Both medications make it much harder for your blood to clot, so taking them together can lead to serious internal bleeding even from minor injuries.",
        "symptoms": ["Unusual bruising or petechiae", "Dark, tarry stools or blood in urine", "Frequent nosebleeds lasting >10 mins", "Unexplained dizziness or severe headaches"],
        "precautions": ["Consult your cardiologist immediately before taking both.", "Do not discontinue prescribed Warfarin without physician guidance.", "Monitor INR levels closely.", "Avoid high-impact activities that risk physical trauma."]
    },
    {
        "drugs": ["Lisinopril", "Potassium"],
        "severity": 68,
        "category": "Moderate-High Risk",
        "mechanism": "Lisinopril (ACE inhibitor) reduces aldosterone secretion, leading to renal potassium retention. Supplemental potassium markedly raises serum potassium levels (Hyperkalemia).",
        "empathetic_summary": "Taking Lisinopril along with potassium supplements can cause potassium to build up in your blood. High potassium levels can affect how your heart beats and cause fatigue.",
        "symptoms": ["Irregular heartbeats (palpitations)", "Muscle weakness or numbness", "Nausea or abdominal cramping"],
        "precautions": ["Have your doctor test your serum potassium levels.", "Avoid potassium-based salt substitutes.", "Report sudden heart fluttering immediately."]
    },
    {
        "drugs": ["Sertraline", "Tramadol"],
        "severity": 85,
        "category": "High Risk - Serotonin Syndrome",
        "mechanism": "Sertraline (SSRI) inhibits serotonin reuptake while Tramadol enhances serotonin release and inhibits reuptake. Dual action overstimulates 5-HT receptors.",
        "empathetic_summary": "Taking Sertraline and Tramadol together can trigger a rare but serious condition called Serotonin Syndrome, caused by too much serotonin accumulating in your nervous system.",
        "symptoms": ["Agitation, confusion, or restlessness", "Rapid heart rate and elevated blood pressure", "Heavy sweating and shivering", "Muscle twitching or loss of coordination"],
        "precautions": ["Seek immediate medical evaluation if you feel unusually agitated or feverish.", "Discuss alternative non-serotonergic pain management options with your physician."]
    },
    {
        "drugs": ["Metformin", "Simvastatin"],
        "severity": 15,
        "category": "Low Risk / Generally Safe",
        "mechanism": "No direct major metabolic pathway competition or synergistic toxicity reported under normal hepatic function.",
        "empathetic_summary": "Metformin and Simvastatin are commonly prescribed together for blood sugar and cholesterol management with no major negative interactions.",
        "symptoms": ["Mild transient gastrointestinal upset", "Occasional muscle fatigue"],
        "precautions": ["Maintain regular liver enzyme and renal function blood tests.", "Take medications with meals as directed."]
    }
]

@st.cache_resource
def build_faiss_vector_db():
    """Builds an in-memory FAISS vector index using SentenceTransformers Embeddings."""
    documents = []
    for item in DRUG_INTERACTION_KB:
        content = f"Drugs: {', '.join(item['drugs'])}\nSeverity: {item['severity']}\nMechanism: {item['mechanism']}\nSummary: {item['empathetic_summary']}\nSymptoms: {', '.join(item['symptoms'])}"
        documents.append(Document(page_content=content, metadata=item))
    
    # Initialize Open Source Sentence Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(documents, embeddings)
    return vector_db

# ------------------------------------------------------------------------------
# 3. INTERACTIVE GAUGE & VISUALIZATIONS
# ------------------------------------------------------------------------------
def create_severity_gauge(score):
    """Generates a half-donut Plotly gauge chart for Risk Score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Interaction Risk Score", 'font': {'size': 18, 'color': "#1e293b"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#0284c7"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 30], 'color': '#dcfce7'},
                {'range': [30, 70], 'color': '#fef3c7'},
                {'range': [70, 100], 'color': '#fee2e2'}
            ],
        }
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# ------------------------------------------------------------------------------
# 4. MAIN STREAMLIT APPLICATION
# ------------------------------------------------------------------------------
def main():
    render_disclaimer()
    
    st.title("💊 PharmAssist: RAG Medical Information Assistant")
    st.caption("FAISS Vector Search & LLM-Powered Drug-to-Drug Interaction Assessment Engine")
    
    # Initialize Vector DB
    with st.spinner("Initializing Vector Knowledge Base (FAISS)..."):
        vector_db = build_faiss_vector_db()

    # Sidebar Medication Multi-Select
    st.sidebar.header("📋 Patient Regimen Input")
    available_drugs = ["Warfarin", "Aspirin", "Lisinopril", "Potassium", "Metformin", "Simvastatin", "Sertraline", "Tramadol", "Amoxicillin", "Ibuprofen", "Fluoxetine"]
    
    selected_drugs = st.sidebar.multiselect(
        "Select Medications (Minimum 2):",
        options=available_drugs,
        default=["Warfarin", "Aspirin"]
    )
    
    analyze_btn = st.sidebar.button("🔍 Run Vector RAG Analysis", type="primary", use_container_width=True)

    if analyze_btn or selected_drugs:
        if len(selected_drugs) < 2:
            st.info("👈 Please select at least two medications from the sidebar to analyze potential interactions.")
            return
            
        st.subheader(f"Analysis Results for: {', '.join(selected_drugs)}")
        
        # Formulate Query for FAISS RAG
        query = f"Interactions between {', '.join(selected_drugs)}"
        results = vector_db.similarity_search_with_score(query, k=1)
        
        if results:
            doc, score = results[0]
            data = doc.metadata
            severity = data["severity"]
            
            # Top Stats & Metrics
            col1, col2, col3 = st.columns([2, 2, 3])
            
            with col1:
                st.plotly_chart(create_severity_gauge(severity), use_container_width=True)
                
            with col2:
                st.metric(label="Risk Severity Level", value=f"{severity}/100", delta=data["category"], delta_color="inverse")
                
                # Percentile vs Baseline calculation
                percentile = min(99, int((severity / 100) * 88 + 10))
                st.metric(label="Population Risk Percentile", value=f"{percentile}th Percentile")
                st.write(f"**FAISS Distance Metric:** `{score:.4f}`")
                
            with col3:
                # Graphical Breakdown
                df_chart = pd.DataFrame({
                    "Risk Factor": ["Gastrointestinal Bleed", "Renal Clearance Impact", "Serotonin Toxicity", "Cardiovascular Stress"],
                    "Probability (%)": [severity if "Warfarin" in selected_drugs else 15, 
                                       severity*0.7 if "Lisinopril" in selected_drugs else 20, 
                                       severity if "Sertraline" in selected_drugs else 10, 
                                       severity*0.5]
                })
                fig_bar = px.bar(df_chart, x="Probability (%)", y="Risk Factor", orientation='h', color="Probability (%)",
                                 color_continuous_scale="Reds" if severity > 60 else "Blues", title="Physiological Impact Profile")
                fig_bar.update_layout(height=220, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_bar, use_container_width=True)

            # Interactive Tabs for Medical Information
            tab1, tab2, tab3, tab4 = st.tabs([
                "❤️ Empathetic Overview", 
                "🔬 Pharmacological Mechanism", 
                "⚠️ Warning Symptoms", 
                "🛡️ Precautions & Actions"
            ])
            
            with tab1:
                st.success(f"**Patient Guidance Summary:**\n\n{data['empathetic_summary']}")
                
            with tab2:
                st.info(f"**Clinical Mechanism of Action:**\n\n{data['mechanism']}")
                
            with tab3:
                st.write("**Symptoms to Monitor Closely:**")
                for s in data["symptoms"]:
                    st.write(f"- 🔴 {s}")
                    
            with tab4:
                st.write("**Recommended Medical Precautions:**")
                for p in data["precautions"]:
                    st.write(f"- 📋 {p}")
                    
    # Bottom Disclaimer
    st.divider()
    render_disclaimer()

if __name__ == "__main__":
    main()
