#!/usr/bin/env python3
"""
Streamlit UI for Hybrid Vector Search Engine

Run with: streamlit run src/ui/streamlit_app.py
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Hybrid Vector Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.stMetric {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None

# Sidebar configuration
st.sidebar.title("🔧 Configuration")
api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")
k = st.sidebar.slider("Top-K Results", 1, 100, 10)
ef = st.sidebar.slider("Search Beam Width (ef)", 1, 500, 50, help="Higher = more accurate, slower")

# Check API health
@st.cache_resource
def get_api_health():
    try:
        resp = requests.get(f"{api_url}/health", timeout=2)
        return resp.status_code == 200
    except:
        return False

@st.cache_data(ttl=10)
def get_index_stats():
    try:
        resp = requests.get(f"{api_url}/stats", timeout=5)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def search_documents(query: str, k: int, ef: int):
    """Search using the API"""
    try:
        resp = requests.post(
            f"{api_url}/search",
            json={"query": query, "k": k, "ef": ef},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Search failed: {e}")
        return None

def ingest_document(text: str, metadata: dict):
    """Ingest a single document"""
    try:
        resp = requests.post(
            f"{api_url}/ingest",
            json={"text": text, "metadata": metadata},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ingestion failed: {e}")
        return None

# Header
st.title("🔍 Hybrid Vector Search Engine")
st.markdown("Fast semantic search using HNSW + sentence-transformers")

# API Status
api_health = get_api_health()
status_col = st.columns([1, 4])[0]
with status_col:
    if api_health:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
        st.info(f"Make sure FastAPI server is running at {api_url}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔎 Search", "📝 Ingest", "📊 Stats", "🧪 Demo"])

# ==================== TAB 1: SEARCH ====================
with tab1:
    st.subheader("Vector Search")
    
    query = st.text_area(
        "Enter your search query:",
        placeholder="e.g., 'what is machine learning?'",
        height=100
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_btn = st.button("🔍 Search", use_container_width=True)
    
    if search_btn and query.strip():
        with st.spinner("Searching..."):
            t0 = time.time()
            results = search_documents(query, k=k, ef=ef)
            elapsed = time.time() - t0
        
        if results:
            # Store results in session state for export
            st.session_state.last_results = results
            st.session_state.last_query = query
            
            # Query info
            st.info(f"🔍 **Query:** {query}")
            
            # Results summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Results Found", len(results.get("hits", [])))
            with col2:
                st.metric("Latency", f"{elapsed*1000:.1f}ms")
            with col3:
                st.metric("Embedding Dim", results.get("embedding_dimension", "N/A"))
            with col4:
                st.metric("Model", results.get("model", "N/A")[:15])
            
            st.divider()
            
            # View options
            view_mode = st.radio("📋 Display Mode:", ["Detailed Cards", "Table View", "JSON"], horizontal=True)
            
            # Export buttons
            export_col1, export_col2, export_col3 = st.columns([1, 1, 4])
            with export_col1:
                if st.button("📥 Export JSON"):
                    json_str = json.dumps(results, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"search_results_{int(time.time())}.json",
                        mime="application/json"
                    )
            with export_col2:
                if st.button("📊 Export CSV"):
                    hits = results.get("hits", [])
                    df = pd.DataFrame([
                        {
                            "Rank": i+1,
                            "ID": hit.get("id", ""),
                            "Score": hit.get("score", 0),
                            "Text": hit.get("text", "")[:100],
                            "Metadata": json.dumps(hit.get("metadata", {}))
                        }
                        for i, hit in enumerate(hits)
                    ])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"search_results_{int(time.time())}.csv",
                        mime="text/csv"
                    )
            
            st.divider()
            
            # Display hits based on selected mode
            if results.get("hits"):
                if view_mode == "Detailed Cards":
                    st.subheader(f"📄 Results ({len(results['hits'])} found)")
                    for i, hit in enumerate(results["hits"], 1):
                        score = hit.get('score', 0)
                        # Normalize score for progress bar (assuming 0-1 range)
                        score_pct = min(100, max(0, score * 100))
                        
                        with st.expander(
                            f"#{i} · **Score: {score:.4f}** · ID: {hit.get('id', 'N/A')}", 
                            expanded=(i==1)
                        ):
                            # Score visualization
                            st.write("**Similarity Score:**")
                            st.progress(score_pct/100.0, text=f"{score:.4f}")
                            
                            # Full text
                            st.write("**Document Text:**")
                            st.text_area(
                                "Content:",
                                value=hit.get('text', 'N/A'),
                                disabled=True,
                                height=150,
                                key=f"text_{i}"
                            )
                            
                            # Metadata
                            if hit.get('metadata'):
                                st.write("**Metadata:**")
                                st.json(hit.get('metadata'))
                
                elif view_mode == "Table View":
                    st.subheader(f"📊 Results Table ({len(results['hits'])} found)")
                    table_data = []
                    for i, hit in enumerate(results["hits"], 1):
                        table_data.append({
                            "Rank": i,
                            "ID": hit.get("id", "N/A"),
                            "Score": f"{hit.get('score', 0):.4f}",
                            "Text Preview": hit.get('text', 'N/A')[:100] + "...",
                            "Full Text": hit.get('text', 'N/A'),
                            "Metadata": json.dumps(hit.get('metadata', {}))
                        })
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                
                else:  # JSON View
                    st.subheader(f"📋 Raw JSON Output ({len(results['hits'])} results)")
                    st.json(results)
            else:
                st.info("No results found")
    elif search_btn:
        st.warning("Please enter a query")

# ==================== TAB 2: INGEST ====================
with tab2:
    st.subheader("Ingest Documents")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Single Document Ingestion**")
        doc_text = st.text_area(
            "Document text:",
            placeholder="Enter document content...",
            height=150,
            key="single_doc"
        )
        
        source = st.text_input("Source (metadata)", placeholder="e.g., 'wiki', 'news'")
        doc_id = st.text_input("Document ID (optional)", placeholder="auto-generated if empty")
        
        if st.button("📥 Ingest", key="ingest_single"):
            if doc_text.strip():
                with st.spinner("Ingesting..."):
                    metadata = {"source": source} if source else {}
                    if doc_id:
                        metadata["custom_id"] = doc_id
                    
                    result = ingest_document(doc_text, metadata)
                    if result:
                        st.success(f"✅ Ingested! ID: {result.get('id', 'N/A')}")
            else:
                st.warning("Please enter document text")
    
    with col2:
        st.write("**Sample Documents**")
        if st.button("Add Sample Docs"):
            samples = [
                {
                    "text": "HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbor algorithm.",
                    "metadata": {"source": "doc", "type": "algorithm"}
                },
                {
                    "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                    "metadata": {"source": "doc", "type": "ml"}
                },
                {
                    "text": "Vector embeddings transform text into high-dimensional numerical representations.",
                    "metadata": {"source": "doc", "type": "embeddings"}
                },
            ]
            
            with st.spinner("Ingesting samples..."):
                for sample in samples:
                    result = ingest_document(sample["text"], sample["metadata"])
                    if result:
                        st.success(f"✅ {sample['text'][:50]}...")
                    else:
                        st.error(f"❌ Failed: {sample['text'][:50]}...")

# ==================== TAB 3: STATS ====================
with tab3:
    st.subheader("Index Statistics")
    
    if st.button("🔄 Refresh Stats"):
        st.cache_data.clear()
    
    stats = get_index_stats()
    
    if "error" not in stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Index Size", stats.get("index_size_mb", "N/A"), "MB")
        with col2:
            st.metric("Embedding Dim", stats.get("embedding_dimension", "N/A"))
        with col3:
            st.metric("Model", stats.get("model", "N/A")[:20])
        with col4:
            st.metric("Status", stats.get("status", "N/A"))
        
        st.divider()
        st.json(stats)
    else:
        st.error(f"Failed to fetch stats: {stats.get('error')}")

# ==================== TAB 4: DEMO ====================
with tab4:
    st.subheader("Quick Demo")
    
    st.write("""
    Try these example queries to explore the search engine:
    """)
    
    demo_queries = [
        ("Machine Learning", "Search for ML-related content"),
        ("Graph Algorithms", "Search for graph-based algorithms"),
        ("Vector Embeddings", "Search for embedding concepts"),
        ("Search Engines", "Search for search engine topics"),
    ]
    
    for i, (query, description) in enumerate(demo_queries):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{query}** — {description}")
        with col2:
            if st.button("Try", key=f"demo_{i}"):
                with st.spinner("Searching..."):
                    results = search_documents(query, k=5, ef=50)
                    if results and results.get("hits"):
                        st.success(f"Found {len(results['hits'])} results!")
                        for hit in results["hits"][:3]:
                            st.caption(f"📄 {hit.get('text', 'N/A')[:80]}...")
                    else:
                        st.info("No results found. Try ingesting sample documents first.")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🚀 Hybrid Vector Search Engine")
with col2:
    st.caption(f"Connected to: {api_url}")
with col3:
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
