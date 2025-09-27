import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Backend imports
from backend.parser import contract_parser
from backend.embeddings import embedding_manager
from backend.db import db_manager
from backend.suggestions import suggestion_generator
from backend.chatbot import contract_chatbot
from config.settings import settings
from backend.risk_scoring import risk_scorer

# Configure Streamlit page
st.set_page_config(
    page_title="Contract Risk Analyzer",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'contracts' not in st.session_state:
    st.session_state.contracts = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = {}

def main():
    """Main application function"""
    st.title("🔍 Contract Risk Analysis Platform")
    st.markdown("Analyze contracts for risk, compliance, and improvement opportunities")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["📤 Upload Contract", "📊 View Analysis", "📈 Risk Dashboard", "💬 Contract Chatbot", "📋 Contract Library"],
        label_visibility="collapsed"
    )
    
    if page == "📤 Upload Contract":
        upload_page()
    elif page == "📊 View Analysis":
        view_analysis_page()
    elif page == "📈 Risk Dashboard":
        risk_dashboard_page()
    elif page == "💬 Contract Chatbot":
        chatbot_page()
    elif page == "📋 Contract Library":
        contract_library_page()

def upload_page():
    """Page for uploading and triggering contract analysis."""
    st.header("Upload Contract for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a contract file",
        type=['pdf', 'docx', 'txt'],
        help="Upload PDF, DOCX, or TXT files"
    )
    
    if uploaded_file is not None:
        st.success(f"File ready for analysis: {uploaded_file.name}")
        
        if st.button("🔍 Analyze Contract", type="primary"):
            with st.spinner("Analyzing contract... This may take a few minutes."):
                analyze_contract(uploaded_file)
            st.success("✅ Analysis complete! Navigate to the 'View Analysis' page to see the detailed results.")
            st.balloons()

def analyze_contract(uploaded_file):
    """Core analysis function, saves results to session state."""
    try:
        contract_text = contract_parser.extract_text(uploaded_file, uploaded_file.name)
        clauses = contract_parser.split_into_clauses(contract_text)
        contract_id = db_manager.insert_contract(uploaded_file.name, contract_text)
        
        clause_analyses = []
        progress_bar = st.progress(0, text="Analyzing clauses...")
        for i, clause in enumerate(clauses):
            embedding = embedding_manager.generate_embedding(clause['text'])
            risk_analysis = risk_scorer.analyze_clause_risk(clause)
            db_manager.insert_clause(
                contract_id, clause['text'], clause['type'],
                risk_analysis['risk_score'], risk_analysis['risk_level'],
                risk_analysis['compliance_status'], embedding
            )
            clause_analyses.append({
                'clause_id': i, 'clause_type': clause['type'], 'clause_text': clause['text'], **risk_analysis
            })
            progress_bar.progress((i + 1) / len(clauses), text=f"Analyzing clause {i+1}/{len(clauses)}")
        
        overall_risk_score, overall_risk_level = risk_scorer.calculate_overall_contract_score(clause_analyses)
        db_manager.update_contract_risk_score(contract_id, overall_risk_score, overall_risk_level)
        
        st.session_state.contracts[contract_id] = {
            'filename': uploaded_file.name,
            'risk_score': overall_risk_score,
            'risk_level': overall_risk_level,
            'clauses': clause_analyses
        }
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")

def view_analysis_page():
    """Page to view detailed analysis of any uploaded contract."""
    st.header("📊 View Detailed Contract Analysis")

    if not st.session_state.contracts:
        st.info("No contracts have been analyzed in this session. Please go to the 'Upload Contract' page to begin.")
        return

    contract_options = {cid: details['filename'] for cid, details in st.session_state.contracts.items()}
    selected_contract_id = st.selectbox(
        "Select a contract to view its analysis:",
        options=list(contract_options.keys()),
        format_func=lambda cid: contract_options[cid]
    )

    if selected_contract_id:
        contract_data = st.session_state.contracts[selected_contract_id]
        display_analysis_results(
            selected_contract_id,
            contract_data['clauses'],
            contract_data['risk_score'],
            contract_data['risk_level']
        )

def display_analysis_results(contract_id, clause_analyses, overall_risk_score, overall_risk_level):
    """Displays the detailed analysis results for a selected contract, showing ALL clauses."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Risk Score", f"{overall_risk_score:.2f}")
    with col2:
        risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
        st.metric("Risk Level", f"{risk_color.get(overall_risk_level, '⚪')} {overall_risk_level}")
    with col3:
        st.metric("Total Clauses", len(clause_analyses))
    with col4:
        high_risk_count = sum(1 for c in clause_analyses if c['risk_level'] == 'HIGH')
        st.metric("High Risk Clauses", high_risk_count)
    
    st.markdown("---")
    
    st.subheader("Complete Clause Analysis")
    risk_colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    
    for clause in clause_analyses:
        risk_emoji = risk_colors.get(clause['risk_level'], '⚪')
        expander_title = f"{risk_emoji} **{clause['clause_type'].replace('_', ' ').title()}** - `{clause['risk_level']}` Risk"
        
        with st.expander(expander_title):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("**Clause Text:**")
                st.text_area("Clause Text", clause['clause_text'], height=150, disabled=True, key=f"text_{contract_id}_{clause['clause_id']}")
                if clause['risk_level'] in ['HIGH', 'MEDIUM']:
                    st.write("**Identified Issues:**")
                    for issue in clause['analysis'].get('issues', []):
                        st.write(f"• {issue}")
                else:
                    st.success("This clause presents a low risk to Hari and Winston Associates LLC.")
            with col2:
                st.metric("Risk Score", f"{clause['risk_score']:.2f}")
                st.metric("Compliance", clause['compliance_status'])
                if clause['risk_level'] in ['HIGH', 'MEDIUM']:
                    clause_id_key = f"{contract_id}_{clause['clause_id']}"
                    if clause_id_key not in st.session_state.suggestions:
                        if st.button("💡 Get Suggestions", key=f"suggest_{clause_id_key}"):
                            with st.spinner("Generating suggestions..."):
                                suggestions = suggestion_generator.generate_clause_suggestions(clause, clause['clause_text'], clause['clause_type'])
                                st.session_state.suggestions[clause_id_key] = suggestions
                                st.rerun()
                    else:
                        suggestions = st.session_state.suggestions[clause_id_key]
                        st.write("**Suggestions:**")
                        for suggestion in suggestions[:3]:
                            with st.container(border=True):
                                st.write(f"**{suggestion.get('priority', 'N/A')} Priority:** {suggestion.get('suggestion', 'N/A')}")
                                if suggestion.get('alternative_wording'):
                                    st.info(f"**Suggested Wording:** {suggestion['alternative_wording']}")
                        if st.button("Hide Suggestions", key=f"hide_{clause_id_key}"):
                            del st.session_state.suggestions[clause_id_key]
                            st.rerun()

def risk_dashboard_page():
    """Risk dashboard and analytics page"""
    st.header("📈 Risk Dashboard")
    
    if not st.session_state.contracts:
        st.info("No contracts analyzed yet. Upload a contract to see the dashboard.")
        return
    
    contract_options = {cid: details['filename'] for cid, details in st.session_state.contracts.items()}
    selected_contract_id = st.selectbox("Select Contract:", options=list(contract_options.keys()), format_func=lambda x: contract_options[x])
    
    if selected_contract_id:
        contract_data = st.session_state.contracts[selected_contract_id]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Risk Score", f"{contract_data['risk_score']:.2f}")
        with col2:
            risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
            st.metric("Risk Level", f"{risk_color[contract_data['risk_level']]} {contract_data['risk_level']}")
        with col3:
            high_risk_count = sum(1 for c in contract_data['clauses'] if c['risk_level'] == 'HIGH')
            st.metric("High Risk Clauses", high_risk_count)
        
        col1, col2 = st.columns(2)
        with col1:
            df_clauses = pd.DataFrame(contract_data['clauses'])
            risk_counts = df_clauses['risk_level'].value_counts()
            fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index, title="Risk Level Distribution", color_discrete_map={'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'})
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            df_clauses = pd.DataFrame(contract_data['clauses'])
            fig_box = px.box(df_clauses, x='clause_type', y='risk_score', title="Risk Score by Clause Type")
            fig_box.update_xaxes(tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)
        
        st.subheader("Contract Improvement Recommendations")
        if st.button("💡 Generate Overall Suggestions", type="primary"):
            with st.spinner("Generating contract-level suggestions..."):
                contract_analysis = {'overall_risk_score': contract_data['risk_score'], 'risk_level': contract_data['risk_level']}
                suggestions = suggestion_generator.generate_contract_level_suggestions(contract_analysis, contract_data['clauses'])
                suggestions = suggestion_generator.prioritize_suggestions(suggestions)
                for suggestion in suggestions:
                    priority_color = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
                    with st.expander(f"{priority_color.get(suggestion.get('priority'), '⚪')} {suggestion.get('priority')} - {suggestion.get('issue', 'N/A')}"):
                        st.write(f"**Suggestion:** {suggestion.get('suggestion', 'No suggestion provided.')}")
                        # --- THIS IS THE FIX ---
                        st.write(f"**Rationale:** {suggestion.get('rationale', 'No rationale provided.')}")

def chatbot_page():
    """Contract chatbot page"""
    st.header("💬 Contract Chatbot")
    st.markdown("Ask questions about your contracts, get explanations, and receive advice.")
    
    if st.session_state.contracts:
        contract_options = {None: "General Questions"}
        contract_options.update({cid: details['filename'] for cid, details in st.session_state.contracts.items()})
        selected_contract_id = st.selectbox("Select Contract Context (optional):", options=list(contract_options.keys()), format_func=lambda x: contract_options[x])
    else:
        selected_contract_id = None
        st.info("Upload a contract to enable contract-specific chat context.")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.write(message['content'])
    
    if user_input := st.chat_input("Ask about your contract..."):
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        st.chat_message("user").write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = contract_chatbot.chat(user_input, selected_contract_id)
                st.write(response['answer'])
                st.session_state.chat_history.append({'role': 'assistant', 'content': response['answer']})

def contract_library_page():
    """Contract library and management page"""
    st.header("📋 Contract Library")
    
    if not st.session_state.contracts:
        st.info("No contracts analyzed in this session. Upload contracts to see them here.")
        return
    
    contract_data = [{'ID': cid, 'Filename': d['filename'], 'Risk Score': f"{d['risk_score']:.2f}", 'Risk Level': d['risk_level'], 'Clauses': len(d['clauses'])} for cid, d in st.session_state.contracts.items()]
    df = pd.DataFrame(contract_data)
    st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()