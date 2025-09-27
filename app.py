import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import json

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
if 'current_contract_id' not in st.session_state:
    st.session_state.current_contract_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def main():
    """Main application function"""
    st.title("🔍 Contract Risk Analysis Platform")
    st.markdown("Analyze contracts for risk, compliance, and improvement opportunities")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📤 Upload & Analyze", "📊 Risk Dashboard", "💬 Contract Chatbot", "📋 Contract Library"]
    )
    
    if page == "📤 Upload & Analyze":
        upload_and_analyze_page()
    elif page == "📊 Risk Dashboard":
        risk_dashboard_page()
    elif page == "💬 Contract Chatbot":
        chatbot_page()
    elif page == "📋 Contract Library":
        contract_library_page()

def upload_and_analyze_page():
    """Upload and analyze contracts page"""
    st.header("Upload and Analyze Contract")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a contract file",
        type=['pdf', 'docx', 'txt'],
        help="Upload PDF, DOCX, or TXT files"
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        if st.button("🔍 Analyze Contract", type="primary"):
            with st.spinner("Analyzing contract... This may take a few minutes."):
                analyze_contract(uploaded_file)

def analyze_contract(uploaded_file):
    """Analyze uploaded contract"""
    try:
        # Extract text
        with st.status("Extracting text from document...") as status:
            contract_text = contract_parser.extract_text(uploaded_file, uploaded_file.name)
            st.write(f"✅ Extracted {len(contract_text)} characters")
            
            # Parse into clauses
            status.update(label="Parsing contract clauses...")
            clauses = contract_parser.split_into_clauses(contract_text)
            st.write(f"✅ Identified {len(clauses)} clauses")
            
            # Save to database
            status.update(label="Saving to database...")
            contract_id = db_manager.insert_contract(uploaded_file.name, contract_text)
            
            # Analyze each clause
            status.update(label="Analyzing clause risks...")
            clause_analyses = []
            
            progress_bar = st.progress(0)
            for i, clause in enumerate(clauses):
                # Generate embedding
                embedding = embedding_manager.generate_embedding(clause['text'])
                
                # Analyze risk
                risk_analysis = risk_scorer.analyze_clause_risk(clause)
                
                # Save clause to database
                db_manager.insert_clause(
                    contract_id, clause['text'], clause['type'],
                    risk_analysis['risk_score'], risk_analysis['risk_level'],
                    risk_analysis['compliance_status'], embedding
                )
                
                clause_analyses.append({
                    'clause_id': i,
                    'clause_type': clause['type'],
                    'clause_text': clause['text'],
                    **risk_analysis
                })
                
                progress_bar.progress((i + 1) / len(clauses))
            
            # Calculate overall risk
            status.update(label="Calculating overall risk score...")
            overall_risk_score, overall_risk_level = risk_scorer.calculate_overall_contract_score(clause_analyses)
            
            # Update contract with overall score
            db_manager.update_contract_risk_score(contract_id, overall_risk_score, overall_risk_level)
            
            status.update(label="Analysis complete!", state="complete")
        
        # Display results
        display_analysis_results(contract_id, clause_analyses, overall_risk_score, overall_risk_level)
        
        # Store in session state
        st.session_state.current_contract_id = contract_id
        st.session_state.contracts[contract_id] = {
            'filename': uploaded_file.name,
            'risk_score': overall_risk_score,
            'risk_level': overall_risk_level,
            'clauses': clause_analyses
        }
        
    except Exception as e:
        st.error(f"Error analyzing contract: {str(e)}")

def display_analysis_results(contract_id, clause_analyses, overall_risk_score, overall_risk_level):
    """Display contract analysis results"""
    st.header("📊 Analysis Results")
    
    # Overall risk metrics
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
    
    # Risk distribution chart
    st.subheader("Risk Distribution by Clause Type")
    
    df_clauses = pd.DataFrame(clause_analyses)
    risk_dist = df_clauses.groupby(['clause_type', 'risk_level']).size().reset_index(name='count')
    
    fig = px.bar(
        risk_dist, x='clause_type', y='count', color='risk_level',
        color_discrete_map={'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'},
        title="Risk Distribution Across Clause Types"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed clause analysis
    st.subheader("Detailed Clause Analysis")
    
    for clause in clause_analyses:
        if clause['risk_level'] in ['HIGH', 'MEDIUM']:  # Show only risky clauses
            with st.expander(f"{clause['clause_type'].title()} - {clause['risk_level']} Risk"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Clause Text:**")
                    st.write(clause['clause_text'][:500] + "..." if len(clause['clause_text']) > 500 else clause['clause_text'])
                    
                    st.write("**Issues:**")
                    for issue in clause['analysis'].get('issues', []):
                        st.write(f"• {issue}")
                    
                    if clause['flags']:
                        st.write("**Risk Flags:**")
                        for flag in clause['flags']:
                            st.write(f"🚩 {flag.replace('_', ' ').title()}")
                
                with col2:
                    st.metric("Risk Score", f"{clause['risk_score']:.2f}")
                    st.metric("Compliance", clause['compliance_status'])
                    
                    # Generate suggestions for this clause
                    if st.button(f"💡 Get Suggestions", key=f"suggest_{clause['clause_id']}"):
                        with st.spinner("Generating suggestions..."):
                            suggestions = suggestion_generator.generate_clause_suggestions(
                                clause, clause['clause_text'], clause['clause_type']
                            )
                            
                            st.write("**Suggestions:**")
                            for i, suggestion in enumerate(suggestions[:3], 1):
                                st.write(f"{i}. **{suggestion['priority']} Priority:** {suggestion['suggestion']}")
                                if suggestion.get('alternative_wording'):
                                    st.info(f"Suggested wording: {suggestion['alternative_wording'][:200]}...")

def risk_dashboard_page():
    """Risk dashboard and analytics page"""
    st.header("📊 Risk Dashboard")
    
    if not st.session_state.contracts:
        st.info("No contracts analyzed yet. Upload a contract to see dashboard.")
        return
    
    # Contract selector
    contract_options = {
        cid: details['filename'] 
        for cid, details in st.session_state.contracts.items()
    }
    
    selected_contract_id = st.selectbox(
        "Select Contract:",
        options=list(contract_options.keys()),
        format_func=lambda x: contract_options[x],
        index=0
    )
    
    if selected_contract_id:
        contract_data = st.session_state.contracts[selected_contract_id]
        
        # Overall metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Risk Score", f"{contract_data['risk_score']:.2f}")
        
        with col2:
            risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
            st.metric("Risk Level", f"{risk_color[contract_data['risk_level']]} {contract_data['risk_level']}")
        
        with col3:
            high_risk_count = sum(1 for c in contract_data['clauses'] if c['risk_level'] == 'HIGH')
            st.metric("High Risk Clauses", high_risk_count)
        
        # Risk breakdown charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk level distribution
            risk_counts = pd.Series([c['risk_level'] for c in contract_data['clauses']]).value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution",
                color_discrete_map={'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Risk scores by clause type
            df_clauses = pd.DataFrame(contract_data['clauses'])
            fig_box = px.box(
                df_clauses,
                x='clause_type',
                y='risk_score',
                title="Risk Score Distribution by Clause Type"
            )
            fig_box.update_xaxes(tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Compliance overview
        st.subheader("Compliance Overview")
        compliance_counts = pd.Series([c['compliance_status'] for c in contract_data['clauses']]).value_counts()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Compliant", compliance_counts.get('COMPLIANT', 0))
        with col2:
            st.metric("Partial", compliance_counts.get('PARTIAL', 0))
        with col3:
            st.metric("Non-Compliant", compliance_counts.get('NON_COMPLIANT', 0))
        
        # Contract-level suggestions
        st.subheader("Contract Improvement Recommendations")
        if st.button("💡 Generate Contract Suggestions", type="primary"):
            with st.spinner("Analyzing contract structure and generating suggestions..."):
                contract_analysis = {
                    'overall_risk_score': contract_data['risk_score'],
                    'risk_level': contract_data['risk_level']
                }
                
                suggestions = suggestion_generator.generate_contract_level_suggestions(
                    contract_analysis, contract_data['clauses']
                )
                
                suggestions = suggestion_generator.prioritize_suggestions(suggestions)
                
                for i, suggestion in enumerate(suggestions, 1):
                    priority_color = {
                        'HIGH': '🔴',
                        'MEDIUM': '🟡',
                        'LOW': '🟢'
                    }
                    
                    with st.expander(f"{priority_color[suggestion['priority']]} {suggestion['priority']} - {suggestion['issue']}"):
                        st.write(f"**Category:** {suggestion['category'].title()}")
                        st.write(f"**Suggestion:** {suggestion['suggestion']}")
                        st.write(f"**Rationale:** {suggestion['rationale']}")
                        
                        if suggestion.get('alternative_wording'):
                            st.info(f"**Suggested Wording:** {suggestion['alternative_wording']}")

def chatbot_page():
    """Contract chatbot page"""
    st.header("💬 Contract Chatbot")
    st.markdown("Ask questions about your contracts, get explanations, and receive advice.")
    
    # Contract selector for context
    if st.session_state.contracts:
        contract_options = {
            None: "General Questions (No specific contract)"
        }
        contract_options.update({
            cid: details['filename'] 
            for cid, details in st.session_state.contracts.items()
        })
        
        selected_contract_id = st.selectbox(
            "Context (optional):",
            options=list(contract_options.keys()),
            format_func=lambda x: contract_options[x],
            index=0
        )
    else:
        selected_contract_id = None
        st.info("Upload a contract first to get contract-specific answers.")
    
    # Chat interface
    st.subheader("Chat")
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            st.chat_message("user").write(message['content'])
        else:
            with st.chat_message("assistant"):
                st.write(message['content'])
                if message.get('sources'):
                    with st.expander("Sources"):
                        for source in message['sources']:
                            st.write(f"- {source['text']} (Risk: {source['risk_level']})")
    
    # Chat input
    user_input = st.chat_input("Ask about your contract...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Display user message
        st.chat_message("user").write(user_input)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = contract_chatbot.chat(user_input, selected_contract_id)
                
                st.write(response['answer'])
                
                # Show sources if available
                if response.get('sources'):
                    with st.expander("Sources"):
                        for source in response['sources']:
                            st.write(f"- {source['text']} (Risk: {source['risk_level']})")
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response['answer'],
                    'sources': response.get('sources', [])
                })
    
    # Chat controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            contract_chatbot.clear_history()
            st.rerun()
    
    with col2:
        if st.button("📄 Get Conversation Summary"):
            if st.session_state.chat_history:
                summary = contract_chatbot.get_conversation_summary()
                st.info(summary)
            else:
                st.warning("No conversation to summarize.")

def contract_library_page():
    """Contract library and management page"""
    st.header("📋 Contract Library")
    
    if not st.session_state.contracts:
        st.info("No contracts in library. Upload contracts to see them here.")
        return
    
    # Library overview
    st.subheader("Library Overview")
    
    total_contracts = len(st.session_state.contracts)
    high_risk_contracts = sum(1 for c in st.session_state.contracts.values() if c['risk_level'] == 'HIGH')
    avg_risk_score = sum(c['risk_score'] for c in st.session_state.contracts.values()) / total_contracts if total_contracts > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Contracts", total_contracts)
    with col2:
        st.metric("High Risk Contracts", high_risk_contracts)
    with col3:
        st.metric("Average Risk Score", f"{avg_risk_score:.2f}")
    
    # Contract list
    st.subheader("Contract List")
    
    contract_data = []
    for contract_id, details in st.session_state.contracts.items():
        contract_data.append({
            'ID': contract_id,
            'Filename': details['filename'],
            'Risk Score': f"{details['risk_score']:.2f}",
            'Risk Level': details['risk_level'],
            'Clauses': len(details['clauses']),
            'High Risk Clauses': sum(1 for c in details['clauses'] if c['risk_level'] == 'HIGH')
        })
    
    df = pd.DataFrame(contract_data)
    
    # Color code the risk levels
    def style_risk_level(val):
        if val == 'HIGH':
            return 'background-color: #ffebee'
        elif val == 'MEDIUM':
            return 'background-color: #fff8e1'
        else:
            return 'background-color: #e8f5e8'
    
    styled_df = df.style.applymap(style_risk_level, subset=['Risk Level'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Contract comparison
    if len(st.session_state.contracts) > 1:
        st.subheader("Contract Comparison")
        
        # Risk score comparison
        risk_scores = [c['risk_score'] for c in st.session_state.contracts.values()]
        filenames = [c['filename'] for c in st.session_state.contracts.values()]
        
        fig_comparison = px.bar(
            x=filenames,
            y=risk_scores,
            title="Risk Score Comparison Across Contracts",
            labels={'x': 'Contract', 'y': 'Risk Score'}
        )
        fig_comparison.update_xaxes(tickangle=45)
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Export functionality
    st.subheader("Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Contract Summary"):
            # Create summary report
            summary_data = []
            for contract_id, details in st.session_state.contracts.items():
                for clause in details['clauses']:
                    summary_data.append({
                        'Contract': details['filename'],
                        'Clause Type': clause['clause_type'],
                        'Risk Level': clause['risk_level'],
                        'Risk Score': clause['risk_score'],
                        'Compliance Status': clause['compliance_status'],
                        'Has Issues': len(clause['analysis'].get('issues', [])) > 0
                    })
            
            df_export = pd.DataFrame(summary_data)
            csv = df_export.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"contract_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Generate Risk Report"):
            # Create detailed risk report
            report_content = "# Contract Risk Analysis Report\n\n"
            report_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            for contract_id, details in st.session_state.contracts.items():
                report_content += f"## {details['filename']}\n"
                report_content += f"- Overall Risk Score: {details['risk_score']:.2f}\n"
                report_content += f"- Risk Level: {details['risk_level']}\n"
                report_content += f"- Total Clauses: {len(details['clauses'])}\n"
                
                high_risk_clauses = [c for c in details['clauses'] if c['risk_level'] == 'HIGH']
                if high_risk_clauses:
                    report_content += f"- High Risk Clauses: {len(high_risk_clauses)}\n"
                    for clause in high_risk_clauses:
                        report_content += f"  - {clause['clause_type'].title()}: {clause['risk_score']:.2f}\n"
                
                report_content += "\n"
            
            st.download_button(
                label="Download Report",
                data=report_content,
                file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check your configuration and try again.")