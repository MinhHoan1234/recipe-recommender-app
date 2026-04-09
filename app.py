import streamlit as st
from src.inference import load_models, get_content_based_recommendations, get_svd_recommendations

# --- UI Setup ---
st.set_page_config(page_title="AI Recipe Recommender", page_icon="👩‍🍳", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.8rem; color: #FF4B4B; text-align: center; font-weight: 800; margin-bottom: 0px; }
    .sub-header { text-align: center; font-size: 1.2rem; color: #666666; margin-bottom: 30px; font-style: italic; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3565/3565418.png", width=120)
    st.write("## About this App")
    st.write("This application acts as your smart kitchen assistant, powered by Machine Learning.")
    st.info("**Models deployed:**\n\n🟢 TF-IDF (Content-Based)\n\n🔵 SVD (Collaborative Filtering)")

st.markdown('<p class="main-header">👩‍🍳 Smart AI Recipe Recommender</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover your next delicious meal based on what you have or what you love!</p>', unsafe_allow_html=True)

# --- Load Models (Imported from src module) ---
try:
    svd_model, tfidf, tfidf_matrix, df_metadata = load_models()
except FileNotFoundError:
    st.error("🚨 Error: 'models' directory not found!")
    st.stop()

# --- Tabs ---
tab1, tab2 = st.tabs(["🥦 Search by Ingredients (For Everyone)", "👥 Search by User ID (For Members)"])

# --- FEATURE 1: CONTENT-BASED SEARCH ---
with tab1:
    st.write("### 🧊 What's in your fridge?")
    ingredient_vocab = tfidf.get_feature_names_out()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_ingredients = st.multiselect("Search and select your ingredients:", options=ingredient_vocab)
    with col2:
        st.write("<br><br>", unsafe_allow_html=True)
        btn_cb = st.button("🔍 Find Recipes", use_container_width=True, type="primary")

    if btn_cb:
        if selected_ingredients:
            with st.spinner("AI is cooking up suggestions..."):
                # CALL PROCESSING FUNCTION FROM INFERENCE MODULE
                results = get_content_based_recommendations(selected_ingredients, tfidf, tfidf_matrix, df_metadata)
                
                if results:
                    st.success(f"Here are the best matches for: {', '.join(selected_ingredients)}")
                    for item in results:
                        with st.expander(f"🍲 {item['name'].title()} (Match: {item['score']:.0%})"):
                            st.write(f"**⏳ Time to make:** {item['minutes']} mins")
                            st.write(f"**🛒 Ingredients:** {', '.join(item['ingredients'])}")
                            st.write(f"**📝 Steps:** {item['steps']}")
                else:
                    st.warning("No perfect matches found. Try selecting different ingredients!")
        else:
            st.warning("⚠️ Please select at least one ingredient!")

# --- FEATURE 2: SVD MODEL ---
with tab2:
    st.write("### 👤 Personalized For You")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_id_input = st.text_input("Enter your User ID:", placeholder="e.g., U9240752")
    with col2:
        st.write("<br>", unsafe_allow_html=True)
        btn_svd = st.button("✨ Get Recommendations", use_container_width=True, type="primary")

    if btn_svd:
        if user_id_input.strip():
            with st.spinner("Analyzing your past ratings..."):
                # CALL PROCESSING FUNCTION FROM INFERENCE MODULE
                results = get_svd_recommendations(user_id_input.strip(), svd_model, df_metadata)
                
                st.success(f"Based on your taste, we highly recommend these 5 recipes:")
                for item in results:
                    with st.expander(f"⭐ {item['predicted_rating']:.2f} Stars - {item['name'].title()}"):
                        st.write(f"**Recipe ID:** {item['recipe_id']}")
                        st.write(f"**⏳ Time to make:** {item['minutes']} mins")
        else:
            st.warning("⚠️ Please enter a User ID first!")