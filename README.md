import os
import faiss
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MedGuard AI | Medication Interaction Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for UI embellishments
st.markdown(
    """
    <style>
    .disclaimer-banner {
        background-color: #FFF3CD;
        color: #856404;
        border-left: 6px solid #FFEBAA;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 25px;
        font-size: 0.95rem;
    }
    .disclaimer-title {
        font-weight: bold;
        color: #856404;
        font-size: 1.05rem;
        margin-bottom: 4px;
    }
    .metric-card {
        background: #F8F9FA;
        border: 1px solid #E9ECEF;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    </style>
",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. KNOWLEDGE BASE & FAISS INDEX INITIALIZATION
# -----------------------------------------------------------------------------
# Detailed clinical drug interaction database
INTERACTION_DATABASE = [
    {
        "drugs": ["Warfarin", "Aspirin"],
        "severity": "High",
        "severity_score": 90,
        "organ_impact": {"Bleeding Risk": 95, "GI Distress": 80, "Renal Strain": 40, "CNS Effect": 20, "Cardiac Strain": 30},
        "description": "Concomitant use of Warfarin and Aspirin significantly elevates major bleeding risk, including gastrointestinal and intracranial hemorrhages, due to dual inhibition of primary and secondary hemostasis."
    },
    {
        "drugs": ["Sertraline", "Tramadol"],
        "severity": "High",
        "severity_score": 88,
        "organ_impact": {"Bleeding Risk": 30, "GI Distress": 50, "Renal Strain": 20, "CNS Effect": 95, "Cardiac Strain": 65},
        "description": "Co-administration increases the risk of Serotonin Syndrome—a potentially life-threatening condition. Both drugs increase central serotonergic transmission through complementary pathways."
    },
    {
        "drugs": ["Lisinopril", "Spironolactone"],
        "severity": "High",
        "severity_score": 85,
        "organ_impact": {"Bleeding Risk": 10, "GI Distress": 30, "Renal Strain": 85, "CNS Effect": 30, "Cardiac Strain": 90},
        "description": "Combining Lisinopril (ACE Inhibitor) and Spironolactone (Potassium-sparing Diuretic) markedly increases the risk of hyperkalemia (dangerously high blood potassium levels), which can cause fatal cardiac arrhythmias."
    },
    {
        "drugs": ["Metformin", "Contrast Dye"],
        "severity": "High",
        "severity_score": 82,
        "organ_impact": {"Bleeding Risk": 10, "GI Distress": 70, "Renal Strain": 95, "CNS Effect": 20, "Cardiac Strain": 40},
        "description": "Iodinated contrast media can cause contrast-induced nephropathy leading to acute renal failure and severe lactic acidosis in patients receiving Metformin."
    },
    {
        "drugs": ["Amoxicillin", "Methotrexate"],
        "severity": "Moderate",
        "severity_score": 65,
        "organ_impact": {"Bleeding Risk": 40, "GI Distress": 60, "Renal Strain": 70, "CNS Effect": 20, "Cardiac Strain": 10},
        "description": "Penicillins like Amoxicillin can decrease the renal clearance of Methotrexate, potentially causing systemic methotrexate toxicity, bone marrow suppression, and severe mucositis."
    },
    {
        "drugs": ["Atorvastatin", "Clarithromycin"],
        "severity": "Moderate",
        "severity_score": 75,
        "organ_impact": {"Bleeding Risk": 10, "GI Distress": 45, "Renal Strain": 60, "CNS Effect": 15, "Cardiac Strain": 50},
        "description": "Clarithromycin is a potent CYP3A4 inhibitor that markedly increases serum concentrations of Atorvastatin, elevating the risk of severe myopathy and rhabdomyolysis."
    },
    {
        "drugs": ["Ibuprofen", "Lisinopril"],
        "severity": "Moderate",
        "severity_score": 60,
        "organ_impact": {"Bleeding Risk": 40, "GI Distress": 65, "Renal Strain": 75, "CNS Effect": 10, "Cardiac Strain": 50},
        "description": "NSAIDs such as Ibuprofen diminish the antihypertensive effect of ACE inhibitors like Lisinopril and increase the risk of acute renal dysfunction due to blunted renal vascular autoregulation."
    },
    {
        "drugs": ["Omeprazole", "Clopidogrel"],
        "severity": "Moderate",
        "severity_score": 58,
        "organ_impact": {"Bleeding Risk": 30, "GI Distress": 30, "Renal Strain": 20, "CNS Effect": 10, "Cardiac Strain": 75},
        "description": "Omeprazole inhibits CYP2C19, preventing the bioactivation of Clopidogrel to its active antiplatelet metabolite, reducing antiplatelet efficacy and increasing ischemic cardiac risk."
    },
    {
        "drugs": ["Acetaminophen", "Alcohol"],
        "severity": "Moderate",
        "severity_score": 70,
        "organ_impact": {"Bleeding Risk": 20, "GI Distress": 50, "Renal Strain": 40, "CNS Effect": 40, "Cardiac Strain": 10},
        "description": "Chronic alcohol consumption induces CYP2E1 enzymes, accelerating conversion of Acetaminophen to NAPQI, a hepatotoxic metabolite, increasing acute liver injury risk."
    },
    {
        "drugs": ["Levothyroxine", "Calcium Carbonate"],
        "severity": "Low",
        "severity_score": 35,
        "organ_impact": {"Bleeding Risk": 0, "GI Distress": 35, "Renal Strain": 10, "CNS Effect": 10, "Cardiac Strain": 10},
        "description": "Calcium Carbonate binds to Levothyroxine in the gastrointestinal tract, forming insoluble complexes that reduce thyroid hormone absorption."
    }
]

@st.cache_resource
def load_faiss_vector_db():
    """Encodes the drug knowledge base and creates a FAISS vector index."""
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    documents = [f"{item['drugs'][0]} and {item['drugs'][1]}: {item['description']}" for item in INTERACTION_DATABASE]
    embeddings = embedder.encode(documents, convert_to_numpy=True)
    
    # Normalize vectors for cosine similarity search
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return embedder, index, INTERACTION_DATABASE

embedder, faiss_index, kb_data = load_faiss_vector_db()

def retrieve_interactions(selected_drugs, k=3):
    """Retrieves verified interactions using FAISS search based on query drugs."""
    query_text = " ".join(selected_drugs)
    query_vector = embedder.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    
    distances, indices = faiss_index.search(query_vector, k)
    retrieved = []
    
    for idx in indices[0]:
        if idx < len(kb_data):
            # Check if any selected drug matches the knowledge base entry
            entry = kb_data[idx]
            match_count = sum(1 for d in selected_drugs if any(d.lower() in drug.lower() or drug.lower() in d.lower() for drug in entry["drugs"]))
            if match_count >= 1:
                retrieved.append(entry)
                
    # Fallback to top vector match if strict filter yields nothing
    if not retrieved and len(indices[0]) > 0:
        retrieved.append(kb_data[indices[0][0]])
        
    return retrieved

# -----------------------------------------------------------------------------
# 3. LLM ANALYSIS GENERATOR
# -----------------------------------------------------------------------------
def generate_llm_explanation(api_key, drugs, retrieved_facts):
    """Uses Gemini 2.5 Flash via google-genai SDK to explain clinical interactions clearly."""
    if not api_key:
        return "⚠️ Please enter a Gemini API key in the sidebar to generate the detailed explanation."
    
    client = genai.Client(api_key=api_key)
    
    context = "\n".join([f"- Interaction: {' + '.join(f['drugs'])} | Severity: {f['severity']} | Detail: {f['description']}" for f in retrieved_facts])
    
    prompt = f"""
    You are an expert clinical pharmacologist and empathetic medical communicator.
    
    Medications entered by user: {', '.join(drugs)}
    Retrieved clinical facts from vector database:
    {context}
    
    Provide a clear, structured, and empathetic medical risk evaluation for the patient using plain English.
    Structure your response using these exact section headings:
    
    ### 1. Medical Risk Overview
    Summarize what happens when these medications interact in terms an average patient can easily understand.
    
    ### 2. Warning Symptoms to Watch Out For
    List 4-5 specific physical or mental symptoms (red flags) that warrant attention. Use bullet points.
    
    ### 3. Immediate Precautions & Recommended Actions
    Give practical, actionable steps the patient should take right now (e.g., timing adjustments, contacting prescribing doctor, dietary caution).
    
    Keep the tone reassuring, professional, and clear. Avoid overly dense jargon without quick explanation.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1000,
            )
        )
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {str(e)}"

# -----------------------------------------------------------------------------
# 4. GRAPHICAL CHART BUILDERS
# -----------------------------------------------------------------------------
def create_gauge_chart(risk_score, severity_label):
    """Generates a Plotly visual gauge chart for interaction severity percentile."""
    color_map = {"Low": "#28A745", "Moderate": "#FFC107", "High": "#DC3545"}
    bar_color = color_map.get(severity_label, "#DC3545")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={'suffix': "%", 'font': {'size': 36}},
        title={'text': f"Interaction Severity Index: <b>{severity_label.upper()}</b>", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#444"},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#CCCCCC",
            'steps': [
                {'range': [0, 39], 'color': '#D4EDDA'},
                {'range': [40, 74], 'color': '#FFF3CD'},
                {'range': [75, 100], 'color': '#F8D7DA'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    fig.update_layout(margin=dict(l=30, r=30, t=50, b=20), height=280)
    return fig

def create_organ_radar_chart(organ_impact):
    """Creates a Radar Chart highlighting organ systems impacted."""
    categories = list(organ_impact.keys())
    values = list(organ_impact.values())
    
    # Close the loop for radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(220, 53, 69, 0.25)',
        line=dict(color='#DC3545', width=2),
        name='Risk Level'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
        ),
        showlegend=False,
        title=dict(text="<b>Organ System Physiological Impact (%)</b>", font=dict(size=15)),
        margin=dict(l=40, r=40, t=50, b=30),
        height=280
    )
    return fig

# -----------------------------------------------------------------------------
# 5. USER INTERFACE & MAIN APP WORKFLOW
# -----------------------------------------------------------------------------

# Medical Emergency & Legal Disclaimer Banner
st.markdown(
    """
    <div class="disclaimer-banner">
        <div class="disclaimer-title">⚠️ Important Medical Disclaimer & Emergency Notice</div>
        This tool is designed strictly for informational and educational purposes using artificial intelligence. 
        It does <b>NOT</b> constitute medical advice, diagnosis, or treatment. 
        <b>If you are experiencing a medical emergency, severe chest pain, extreme bleeding, or difficulty breathing, call emergency services (911/112) immediately.</b>
        Always consult a licensed physician or pharmacist before altering any drug regimen.
    </div>
    """,
    unsafe_allow_html=True
)

st.title("💊 MedGuard AI: Drug Interaction Assistant")
st.write("FAISS vector search indexing with LLM clinical safety synthesis.")

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")
api_key_input = st.sidebar.text_input(
    "Gemini API Key", 
    value=os.getenv("GEMINI_API_KEY", ""), 
    type="password",
    help="Required for generating plain-English medical explanations."
)

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Presets for Quick Testing")
preset = st.sidebar.selectbox(
    "Select an example pair:",
    ["Custom Input", "Warfarin + Aspirin", "Sertraline + Tramadol", "Lisinopril + Spironolactone", "Atorvastatin + Clarithromycin"]
)

# Manage Drug Input Selection
preset_mapping = {
    "Warfarin + Aspirin": ["Warfarin", "Aspirin"],
    "Sertraline + Tramadol": ["Sertraline", "Tramadol"],
    "Lisinopril + Spironolactone": ["Lisinopril", "Spironolactone"],
    "Atorvastatin + Clarithromycin": ["Atorvastatin", "Clarithromycin"]
}

default_drugs = preset_mapping.get(preset, ["Warfarin", "Aspirin"])

# Multi-select input for medication entry
available_drugs = sorted(list(set([d for item in INTERACTION_DATABASE for d in item["drugs"]] + ["Warfarin", "Aspirin", "Sertraline", "Tramadol", "Lisinopril", "Spironolactone", "Metformin", "Atorvastatin", "Clarithromycin", "Ibuprofen", "Omeprazole", "Clopidogrel", "Acetaminophen", "Levothyroxine"])))

selected_drugs = st.multiselect(
    "Enter or select two or more medications to analyze:",
    options=available_drugs,
    default=default_drugs,
    help="Add at least 2 medications to run similarity indexing."
)

if len(selected_drugs) < 2:
    st.warning("👉 Please select at least two medications to evaluate interactions.")
else:
    if st.button("🔍 Analyze Drug Interactions", type="primary"):
        with st.spinner("Searching FAISS vector database and generating clinical assessment..."):
            
            # 1. FAISS Search Retrieval
            matches = retrieve_interactions(selected_drugs)
            
            if matches:
                primary_match = matches[0]
                risk_score = primary_match["severity_score"]
                severity = primary_match["severity"]
                organ_impact = primary_match["organ_impact"]
            else:
                risk_score = 15
                severity = "Low"
                organ_impact = {"Bleeding Risk": 10, "GI Distress": 10, "Renal Strain": 10, "CNS Effect": 10, "Cardiac Strain": 10}
                primary_match = {
                    "drugs": selected_drugs,
                    "severity": "Low",
                    "severity_score": 15,
                    "description": "No severe high-risk vector match found in primary knowledge base. Standard caution applies."
                }

            # 2. Key Graphical Metrics & Gauges
            col1, col2 = st.columns([1, 1])
            
            with col1:
                gauge_fig = create_gauge_chart(risk_score, severity)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
            with col2:
                radar_fig = create_organ_radar_chart(organ_impact)
                st.plotly_chart(radar_fig, use_container_width=True)
                
            st.markdown("---")
            
            # 3. Vector Match Details & Explanation
            col_left, col_right = st.columns([1, 1.2])
            
            with col_left:
                st.subheader("📋 Vector Index Search Results")
                for idx, m in enumerate(matches, 1):
                    badge_color = "red" if m['severity'] == "High" else "orange" if m['severity'] == "Moderate" else "green"
                    st.markdown(f"**Match #{idx}: { ' + '.join(m['drugs'])}**")
                    st.markdown(f"Severity Grade: **:{badge_color}[{m['severity']} ({m['severity_score']}% Risk Index)]**")
                    st.info(m['description'])
            
            with col_right:
                st.subheader("🤖 AI Clinical Explanation & Guidance")
                explanation = generate_llm_explanation(api_key_input, selected_drugs, matches)
                st.markdown(explanation)

# -----------------------------------------------------------------------------
# 6. KNOWLEDGE BASE SEARCH EXPLORER
# -----------------------------------------------------------------------------
with st.expander("📚 Explore Full Vector Knowledge Base"):
    kb_df = pd.DataFrame([
        {
            "Medication Pair": " + ".join(item["drugs"]),
            "Severity Level": item["severity"],
            "Risk Score (%)": item["severity_score"],
            "Clinical Description": item["description"]
        }
        for item in INTERACTION_DATABASE
    ])
    st.dataframe(kb_df, use_container_width=True)
