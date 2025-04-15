import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import time
import base64
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="PATHFINDER//",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS with enhanced styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #d2ff1e, #1a1a1a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        padding-top: 20px;
        animation: fadeIn 1.5s;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #d2ff1e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
        animation: slideInRight 0.5s;
    }
    
    .result-card {
        background-color: #1a1a1a;
        border-radius: 16px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        border-left: 6px solid #d2ff1e;
        animation: fadeIn 0.8s;
        color: #ffffff;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
    
    .skill-chip {
        background-color: rgba(210, 255, 30, 0.1);
        border-radius: 24px;
        padding: 8px 16px;
        margin: 4px;
        display: inline-block;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
        border: 1px solid #d2ff1e;
        color: #d2ff1e;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .skill-chip:hover {
        transform: scale(1.05);
        box-shadow: 0 3px 8px rgba(210, 255, 30, 0.2);
        background-color: rgba(210, 255, 30, 0.2);
    }
    
    .missing-skill-chip {
        background-color: rgba(210, 255, 30, 0.05);
        border-radius: 24px;
        padding: 8px 16px;
        margin: 4px;
        display: inline-block;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
        border: 1px solid #d2ff1e;
        color: #d2ff1e;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .missing-skill-chip:hover {
        transform: scale(1.05);
        box-shadow: 0 3px 8px rgba(210, 255, 30, 0.2);
        background-color: rgba(210, 255, 30, 0.1);
    }
    
    .profile-card {
        background-color: #1a1a1a;
        border-radius: 16px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        border-top: 6px solid #d2ff1e;
        animation: fadeIn 0.8s;
        color: #ffffff;
    }
    
    .profile-section {
        margin-bottom: 25px;
        padding: 20px;
        background-color: #1a1a1a;
        border-radius: 12px;
        border: 1px solid #d2ff1e;
    }
    
    .profile-section h3 {
        color: #d2ff1e;
        font-size: 1.4rem;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .stButton > button {
        background-color: #1a1a1a;
        color: #d2ff1e;
        font-weight: 600;
        border-radius: 30px;
        padding: 12px 28px;
        border: 2px solid #d2ff1e;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(210, 255, 30, 0.2);
        display: block;
        margin: 20px auto;
        width: 100%;
        max-width: 300px;
        font-size: 1rem;
    }
    
    .stButton > button:hover {
        background-color: #d2ff1e;
        color: #1a1a1a;
        box-shadow: 0 6px 15px rgba(210, 255, 30, 0.3);
        transform: translateY(-2px);
    }
    
    /* Progress tracker */
    .progress-step {
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        margin: 5px;
        transition: all 0.3s ease;
    }
    
    .progress-step.active {
        background-color: rgba(210, 255, 30, 0.1);
        color: #d2ff1e;
        font-weight: 600;
    }
    
    .progress-step.completed {
        color: #d2ff1e;
    }
    
    .progress-step.pending {
        color: #666666;
    }
    
    /* Animation keyframes */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 16px;
        background-color: #1a1a1a;
        border-top: 1px solid #d2ff1e;
        color: #d2ff1e;
        font-size: 0.9rem;
        z-index: 1000;
    }

    /* Update the tech background */
    .tech-background {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
    }

    /* Update the introduction card */
    .intro-card {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        margin-bottom: 25px;
        border-left: 4px solid #d2ff1e;
        animation: fadeIn 1s;
        color: #ffffff;
    }

    /* Update the warning message */
    .stWarning {
        background-color: rgba(210, 255, 30, 0.1);
        border: 1px solid #d2ff1e;
        color: #d2ff1e;
    }

    /* Update the success message */
    .stSuccess {
        background-color: rgba(210, 255, 30, 0.1);
        border: 1px solid #d2ff1e;
        color: #d2ff1e;
    }

    /* Add custom CSS for video links */
    .video-link {
        display: inline-block;
        color: #d2ff1e;
        text-decoration: none;
        font-weight: 500;
        margin: 5px 0;
        padding: 8px 15px;
        border: 1px solid #d2ff1e;
        border-radius: 20px;
        background-color: rgba(210, 255, 30, 0.1);
        transition: all 0.3s ease;
    }
    .video-link:hover {
        background-color: rgba(210, 255, 30, 0.2);
        transform: translateX(5px);
    }
    .skill-card {
        margin: 15px 0;
        padding: 20px;
        background-color: #1a1a1a;
        border-radius: 12px;
        border: 1px solid #d2ff1e;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Function to create pulsating effect on load
def load_animation():
    with st.spinner(''):
        progress_text = "Loading your career recommendation system..."
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        st.success("Ready! Let's find your ideal career path.")
        time.sleep(0.5)
        progress_bar.empty()

# Run initial loading animation only once
if 'loaded' not in st.session_state:
    load_animation()
    st.session_state.loaded = True

# Generate a tech background SVG for the header
def get_tech_background():
    svg = '''
    <svg width="1200" height="150" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#1E88E5;stop-opacity:0.2" />
                <stop offset="100%" style="stop-color:#0D47A1;stop-opacity:0.1" />
            </linearGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#grad1)" />
        <g fill="#1E88E5" opacity="0.1">
            <text x="50" y="50" font-family="monospace" font-size="20">&lt;/&gt;</text>
            <text x="120" y="80" font-family="monospace" font-size="15">{code}</text>
            <text x="200" y="40" font-family="monospace" font-size="18">01010</text>
            <text x="300" y="70" font-family="monospace" font-size="22">function()</text>
            <text x="450" y="50" font-family="monospace" font-size="20">API</text>
            <text x="500" y="90" font-family="monospace" font-size="16">class</text>
            <text x="600" y="60" font-family="monospace" font-size="25">[]</text>
            <text x="650" y="40" font-family="monospace" font-size="18">#include</text>
            <text x="750" y="80" font-family="monospace" font-size="20">import</text>
            <text x="850" y="50" font-family="monospace" font-size="15">python</text>
            <text x="920" y="70" font-family="monospace" font-size="18">java</text>
            <text x="1000" y="40" font-family="monospace" font-size="20">AI</text>
            <text x="1050" y="90" font-family="monospace" font-size="16">ML</text>
        </g>
    </svg>
    '''
    b64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    return f'data:image/svg+xml;base64,{b64}'

# Create a container with the tech background
st.markdown(f'''
    <div style="position: relative; height: 150px; margin-bottom: 30px;">
        <img src="{get_tech_background()}" style="width: 100%; height: 100%; position: absolute; top: 0; left: 0; z-index: 1;">
        <div style="position: relative; z-index: 2; width: 100%; height: 100%; display: flex; justify-content: center; align-items: center;">
            <h1 class="main-header">PATHFINDER//</h1>
        </div>
    </div>
''', unsafe_allow_html=True)

# Introduction with nicer formatting
st.markdown("""
<div class="intro-card">
    <h3 style="color: #d2ff1e; margin-bottom: 10px;">Welcome to Your Career Path Finder</h3>
    <p style="font-size: 16px; line-height: 1.6; color: #ffffff;">
        This intelligent system helps Computer Science and Engineering students discover their ideal career paths based on their unique educational background, skills, and interests. Using advanced machine learning, we analyze your profile and provide personalized recommendations, along with identifying skill gaps to help you prepare for your dream career.
    </p>
    <p style="font-size: 16px; line-height: 1.6; margin-top: 10px; color: #ffffff;">
        <strong style="color: #d2ff1e;">How it works:</strong> Fill in your profile details, click the recommendation button, and instantly receive personalized career insights tailored just for you!
    </p>
</div>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    try:
        # First try to load from uploaded file if there is one
        if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
            df = pd.read_csv(st.session_state.uploaded_file)
        else:
            # Otherwise use the default path
            df = pd.read_csv("cse_career_recommendation_dataset_freshers.csv")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Sidebar for file upload
with st.sidebar:
    st.header("Data Settings")
    uploaded_file = st.file_uploader("Upload your own dataset (optional)", type=["csv"])
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.success("File uploaded successfully!")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses machine learning to recommend career paths for 
    Computer Science and Engineering students based on their profile.
    """)
    st.markdown("---")
   

# Load the data
df = load_data()

if df is not None:
    # Preprocess
    def split_and_strip(series):
        return series.str.split(',').apply(lambda x: [s.strip() for s in x])
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    df['Skills'] = split_and_strip(df['Skills'])
    df['Interests'] = split_and_strip(df['Interests'])
    
    # Extract unique values for dropdowns
    degrees = sorted(df['Degree'].unique())
    courses = sorted(df['Course'].unique())
    specializations = sorted(df['Specialization'].unique())
    
    # Extract all unique skills and interests
    all_skills = sorted(list(set([skill for skills in df['Skills'] for skill in skills])))
    all_interests = sorted(list(set([interest for interests in df['Interests'] for interest in interests])))
    
    # One-hot encode categorical columns
    ohe = OneHotEncoder(sparse_output=False)
    cat_features = ohe.fit_transform(df[['Degree', 'Course', 'Specialization']])
    
    # Multi-hot encode skills and interests
    mlb_skills = MultiLabelBinarizer()
    mlb_interests = MultiLabelBinarizer()
    skills_encoded = mlb_skills.fit_transform(df['Skills'])
    interests_encoded = mlb_interests.fit_transform(df['Interests'])
    
    # Combine all features
    X = np.hstack([cat_features, skills_encoded, interests_encoded])
    
    # Encode the target
    le = LabelEncoder()
    y = le.fit_transform(df['Recommended_Career_Path'])
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    @st.cache_resource
    def train_model():
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    model = train_model()
    
    # Function to get top skills for a career
    def get_top_skills_for_career(career, top_n=10):
        career_rows = df[df['Recommended_Career_Path'] == career]
        all_skills = [skill for skills in career_rows['Skills'] for skill in skills]
        common_skills = Counter(all_skills).most_common(top_n)
        return [skill for skill, count in common_skills]
    
    # Function to predict top careers and identify skill gaps
    def predict_top_careers_and_gaps(user_input, top_k=3):
        # Prepare categorical features
        df_cat = pd.DataFrame([{
            "Degree": user_input["Degree"],
            "Course": user_input["Course"],
            "Specialization": user_input["Specialization"]
        }])
        cat_input = ohe.transform(df_cat)
        
        # Prepare skills and interests features
        input_skills = [s for s in user_input["Skills"] if s in mlb_skills.classes_]
        input_interests = [i for i in user_input["Interests"] if i in mlb_interests.classes_]
        
        skills_input = mlb_skills.transform([input_skills])
        interests_input = mlb_interests.transform([input_interests])
        
        # Combine features
        x_input = np.hstack([cat_input, skills_input, interests_input])
        
        # Get probabilities for all career paths
        probas = model.predict_proba(x_input)[0]
        
        # Get top k career indices
        top_indices = np.argsort(probas)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            career = le.inverse_transform([idx])[0]
            match_score = round(probas[idx] * 100, 2)
            
            # Get required skills and missing skills
            required_skills = get_top_skills_for_career(career, top_n=10)
            missing_skills = list(set(required_skills) - set(input_skills))
            
            results.append({
                "career": career,
                "match_percent": match_score,
                "required_skills": required_skills,
                "missing_skills": missing_skills
            })
        
        return results
    
    # Tabs for different sections
    tab1, tab2 = st.tabs(["Career Recommendation", "Dataset Insights"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>Your Profile</h2>", unsafe_allow_html=True)
        
        # Create layout with columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Educational Background
            st.markdown("### Educational Background")
            degree = st.selectbox("Degree", options=degrees)
            course = st.selectbox("Course", options=courses)
            specialization = st.selectbox("Specialization", options=specializations)
        
        with col2:
            # Skills and Interests
            st.markdown("### Skills and Interests")
            skills = st.multiselect("Skills", options=all_skills)
            interests = st.multiselect("Interests", options=all_interests)
        
        # Button to get recommendations
        if st.button("Get Career Recommendations", type="primary"):
            if not skills or not interests:
                st.warning("Please select at least one skill and one interest to get recommendations.")
            else:
                # User input
                user_input = {
                    "Degree": degree,
                    "Course": course,
                    "Specialization": specialization,
                    "Skills": skills,
                    "Interests": interests
                }
                
                # Get recommendations
                recommendations = predict_top_careers_and_gaps(user_input)
                
                st.markdown("<h2 class='sub-header'>Career Recommendations</h2>", unsafe_allow_html=True)
                
                # Display recommendations
                for i, rec in enumerate(recommendations):
                    # Create a card-like UI for each recommendation
                    st.markdown(f"""
                    <div class='result-card'>
                        <h3>{i+1}. {rec['career']} - {rec['match_percent']}% Match</h3>
                        <div style='margin-top: 10px; margin-bottom: 15px;'>
                            <strong>Required Skills:</strong><br/>
                            {''.join(['<span class="skill-chip">' + skill + '</span>' for skill in rec['required_skills']])}
                        </div>
                        <div>
                            <strong>{'Skills to Develop:' if rec['missing_skills'] else 'You have all the required skills!'}</strong><br/>
                            {''.join(['<span class="missing-skill-chip">' + skill + '</span>' for skill in rec['missing_skills']])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add spacing between cards
                    st.markdown("<br/>", unsafe_allow_html=True)
                
                # Create a bar chart visualization of the match percentages
                st.markdown("<h3>Match Percentage Comparison</h3>", unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                careers = [rec['career'] for rec in recommendations]
                match_percentages = [rec['match_percent'] for rec in recommendations]
                
                # Create bar chart with gradient colors
                bars = ax.barh(careers, match_percentages, color=['#d2ff1e', '#a8e600', '#8fcc00'])
                
                # Add percentage labels
                for i, bar in enumerate(bars):
                    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                            f"{match_percentages[i]}%", va='center', color='#ffffff')
                
                ax.set_xlabel('Match Percentage (%)', color='#ffffff')
                ax.set_xlim(0, 110)  # Add some space for labels
                ax.grid(axis='x', linestyle='--', alpha=0.7, color='#d2ff1e')
                
                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#d2ff1e')
                ax.spines['left'].set_color('#d2ff1e')
                ax.tick_params(axis='x', colors='#ffffff')
                ax.tick_params(axis='y', colors='#ffffff')
                
                # Set background color
                ax.set_facecolor('#1a1a1a')
                fig.patch.set_facecolor('#1a1a1a')
                
                st.pyplot(fig)
                
                # Load video recommendations
                @st.cache_data
                def load_video_recommendations():
                    try:
                        return pd.read_csv("skill_videos.csv")
                    except Exception as e:
                        st.error(f"Error loading video recommendations: {e}")
                        return None
                
                # Provide a skill development plan
                st.markdown("<h3 style='color: #d2ff1e; margin-top: 30px;'>Skill Development Plan</h3>", unsafe_allow_html=True)
                
                # Get all missing skills across recommendations
                all_missing_skills = set()
                for rec in recommendations:
                    all_missing_skills.update(rec['missing_skills'])
                
                if all_missing_skills:
                    st.markdown("""
                    <div style='background-color: #1a1a1a; padding: 20px; border-radius: 12px; border: 1px solid #d2ff1e; margin-bottom: 20px;'>
                        <p style='color: #ffffff; font-size: 16px; margin-bottom: 15px;'>
                            To improve your career prospects, consider developing these skills:
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Load video recommendations
                    video_recommendations = load_video_recommendations()
                    
                    for skill in all_missing_skills:
                        # Find which careers need this skill
                        relevant_careers = [rec['career'] for rec in recommendations if skill in rec['missing_skills']]
                        career_str = ", ".join(relevant_careers)
                        
                        # Find video recommendations for this skill
                        skill_videos = []
                        if video_recommendations is not None:
                            skill_videos = video_recommendations[video_recommendations['skill'].str.lower() == skill.lower()]
                        
                        # Generate video recommendations HTML
                        video_html = ""
                        if not skill_videos.empty:
                            for _, row in skill_videos.iterrows():
                                st.markdown(f"""
                                <div class="skill-card">
                                    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
                                        <span class="missing-skill-chip">{skill}</span>
                                        <div>
                                            <p style="color: #ffffff; margin: 0; font-size: 14px;">
                                                <strong style="color: #d2ff1e;">Needed for:</strong> {career_str}
                                            </p>
                                        </div>
                                    </div>
                                    <div>
                                        <h4 style="color: #d2ff1e; margin-bottom: 10px;">Recommended Learning Resources:</h4>
                                        <a href="{row['video_link']}" target="_blank" class="video-link">
                                            ▶ {row['video_title']}
                                        </a>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="skill-card">
                                <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
                                    <span class="missing-skill-chip">{skill}</span>
                                    <div>
                                        <p style="color: #ffffff; margin: 0; font-size: 14px;">
                                            <strong style="color: #d2ff1e;">Needed for:</strong> {career_str}
                                        </p>
                                    </div>
                                </div>
                                <div>
                                    <h4 style="color: #d2ff1e; margin-bottom: 10px;">Recommended Learning Resources:</h4>
                                    <a href="https://www.youtube.com/results?search_query={skill}+tutorial" 
                                       target="_blank" 
                                       class="video-link">
                                        ▶ Search for "{skill} tutorial" on YouTube
                                    </a>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='
                        background-color: #1a1a1a; 
                        padding: 30px; 
                        border-radius: 12px; 
                        border: 1px solid #d2ff1e;
                        text-align: center;
                        margin-top: 20px;
                    '>
                        <h4 style='color: #d2ff1e; margin-bottom: 15px;'>🎉 Great Job!</h4>
                        <p style='color: #ffffff; font-size: 16px; margin: 0;'>
                            You have all the key skills needed for your recommended career paths.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<h2 class='sub-header'>Dataset Insights</h2>", unsafe_allow_html=True)
        
        # Extract career paths
        career_paths = sorted(df['Recommended_Career_Path'].unique())
        
        # Create charts for dataset insights
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of Career Paths
            st.markdown("### Career Path Distribution")
            
            career_counts = df['Recommended_Career_Path'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            career_counts.plot(kind='barh', ax=ax, color='#d2ff1e')
            ax.set_title('Number of Profiles per Career Path', color='#ffffff')
            ax.set_xlabel('Count', color='#ffffff')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Distribution of Degrees
            st.markdown("### Degree Distribution")
            
            degree_counts = df['Degree'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            degree_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=['#d2ff1e', '#a8e600', '#8fcc00', '#76b300', '#5d9900'])
            ax.set_title('Distribution of Degrees', color='#ffffff')
            ax.set_ylabel('', color='#ffffff')
            
            # Set background color
            ax.set_facecolor('#1a1a1a')
            fig.patch.set_facecolor('#1a1a1a')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Skills frequency
        st.markdown("### Most Common Skills in the Dataset")
        
        # Count skill frequencies across all profiles
        skill_counter = Counter([skill for skills in df['Skills'] for skill in skills])
        top_skills = skill_counter.most_common(20)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        skill_names = [skill[0] for skill in top_skills]
        skill_freqs = [skill[1] for skill in top_skills]
        
        bars = ax.barh(skill_names, skill_freqs, color='#d2ff1e')
        
        # Add count labels
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f"{skill_freqs[i]}", va='center', color='#ffffff')
        
        ax.set_title('Top 20 Skills by Frequency', color='#ffffff')
        ax.set_xlabel('Count', color='#ffffff')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#d2ff1e')
        ax.spines['left'].set_color('#d2ff1e')
        ax.tick_params(axis='x', colors='#ffffff')
        ax.tick_params(axis='y', colors='#ffffff')
        
        # Set background color
        ax.set_facecolor('#1a1a1a')
        fig.patch.set_facecolor('#1a1a1a')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Career Explorer
        st.markdown("<h3 class='sub-header'>Career Explorer</h3>", unsafe_allow_html=True)
        
        selected_career = st.selectbox("Select a career to explore:", options=career_paths)
        
        if selected_career:
            # Filter data for the selected career
            career_data = df[df['Recommended_Career_Path'] == selected_career]
            
            # Get top skills for this career
            career_skills = get_top_skills_for_career(selected_career, top_n=15)
            
            st.markdown(f"### Top Skills for {selected_career}")
            
            # Create visual representation of skills
            skill_html = ""
            for skill in career_skills:
                skill_html += f'<span class="skill-chip">{skill}</span> '
            
            st.markdown(f"<div style='margin-top: 10px;'>{skill_html}</div>", unsafe_allow_html=True)
            
            # Get distribution of degrees for this career
            st.markdown(f"### Educational Background for {selected_career}")
            
            degree_dist = career_data['Degree'].value_counts()
            specialization_dist = career_data['Specialization'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Degree distribution for this career
                fig, ax = plt.subplots(figsize=(8, 5))
                degree_dist.plot(kind='pie', ax=ax, autopct='%1.1f%%', 
                                colors=['#d2ff1e', '#a8e600', '#8fcc00', '#76b300', '#5d9900'])
                ax.set_title('Degree Distribution', color='#ffffff')
                ax.set_ylabel('', color='#ffffff')
                
                # Set background color
                ax.set_facecolor('#1a1a1a')
                fig.patch.set_facecolor('#1a1a1a')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Specialization distribution for this career
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # If there are too many specializations, limit to top 5
                if len(specialization_dist) > 5:
                    top_specs = specialization_dist.head(5)
                    others = pd.Series({'Others': specialization_dist[5:].sum()})
                    spec_dist = pd.concat([top_specs, others])
                else:
                    spec_dist = specialization_dist
                
                spec_dist.plot(kind='pie', ax=ax, autopct='%1.1f%%', 
                              colors=['#d2ff1e', '#a8e600', '#8fcc00', '#76b300', '#5d9900'])
                ax.set_title('Specialization Distribution', color='#ffffff')
                ax.set_ylabel('', color='#ffffff')
                
                # Set background color
                ax.set_facecolor('#1a1a1a')
                fig.patch.set_facecolor('#1a1a1a')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Interests associated with this career
            st.markdown(f"### Common Interests for {selected_career}")
            
            # Count interest frequencies for this career
            interest_counter = Counter([interest for interests in career_data['Interests'] for interest in interests])
            top_interests = interest_counter.most_common(10)
            
            interest_html = ""
            for interest, count in top_interests:
                interest_html += f'<span class="skill-chip">{interest} ({count})</span> '
            
            st.markdown(f"<div style='margin-top: 10px;'>{interest_html}</div>", unsafe_allow_html=True)
else:
    st.error("Failed to load the dataset. Please check the file path or upload a compatible CSV file.")

# Create a progress tracker
steps = ["Education", "Skills", "Interests"]
current_step = st.session_state.get('current_step', 0)

# Progress bar
st.progress((current_step + 1) / len(steps))

# Step indicators
cols = st.columns(len(steps))
for i, step in enumerate(steps):
    with cols[i]:
        if i < current_step:
            st.markdown(f"<div class='progress-step completed'>✓ {step}</div>", unsafe_allow_html=True)
        elif i == current_step:
            st.markdown(f"<div class='progress-step active'>• {step}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='progress-step pending'>○ {step}</div>", unsafe_allow_html=True)

# Add footer
st.markdown("""
<div class="footer">
    <p>© PATHFINDER ML MODELS~ BY PRABHATH & MOHIT</p>
</div>
""", unsafe_allow_html=True)