import pandas as pd 
import numpy as np
import streamlit as st
import plotly.express as px
import pickle
from PIL import Image
import re
from CHATBOT import run_chatbot
import requests
import plotly.graph_objects as go


# Load your datasets
df = pd.read_csv("cleaned_data.csv")
df1 = pd.read_csv("final_data.csv")


# Function to safely convert to sqrt
def safe_sqrt(value):
    try:
        return np.sqrt(float(value))  # Convert to float and take sqrt
    except (ValueError, TypeError):
        return np.nan  
    
# Define occupation types in alphabetical order with corresponding numeric codes
occupation_types = {
    0: 'Accountants',
    1: 'Cleaning staff',
    2: 'Cooking staff',
    3: 'Core staff',
    4: 'Drivers',
    5: 'HR staff',
    6: 'High skill tech staff',
    7: 'IT staff',
    8: 'Laborers',
    9: 'Low-skill Laborers',
    10: 'Managers',
    11: 'Medicine staff',
    12: 'Private service staff',
    13: 'Realty agents',
    14: 'Sales staff',
    15: 'Secretaries',
    16: 'Security staff',
    17: 'Waiters/barmen staff',
}


# Mapping for NAME_EDUCATION_TYPE
education_type_mapping = {'Secondary / secondary special': 0, 'Higher education': 1, 'Incomplete higher': 2, 'Lower secondary': 3, 'Academic degree': 4}
housing_type_mapping = {'Co-op apartment': 0, 'House/apartment': 1, 'Municipal Apartment': 2, 'Office apartment': 3, 'Rented apartment': 4,'With parents':5}
gender_mapping = {'F': 0, 'M': 1, 'XNA': 2}
own_car_mapping = {'N': 0, 'Y': 1,}
# Mapping for NAME_FAMILY_STATUS
family_status_mapping = {'Single / not married': 3, 'Married': 1, 'Civil marriage': 0, 'Widow': 4, 'Separated': 2}

# Main Streamlit code
# -------------------------------------------------- Logo & details on top
# Custom CSS to style the sidebar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 10px;
    }
    .sidebar .sidebar-content .element-container {
        margin-bottom: 15px;
    }
    .sidebar .sidebar-content .element-container .stButton button {
        background-color: #ff4b4b;
        color: blue;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
with st.sidebar:
  st.image("pngwing.com.png", width=150)

st.sidebar.title("Bank Risk")
page = st.sidebar.radio("Go to", ["Home", "Data Showcase", "ML Prediction", "Data Visualization", "About", "CHATBOT"])

# Sidebar user information and logout button
st.sidebar.write("User: Srinikesh Sathiaseelan")

# Define tab options
#tabs = ["Home", "Data Showcase", "ML Prediction", "ChatBot", "Data Visualization", "About"]
#selected_tab = st.sidebar.selectbox("Select a tab", tabs)

if page != "CHATBOT":
    with st.sidebar:
        st.image("—Pngtree—a robot chatbot_15922733.png", width=50)
    st.image("core-banking-solutions-scaled.webp", caption="Bank Risk Controller")
    st.markdown("# :red[*Bank*] *Risk* :red[*Controller*] *System*")
    st.markdown("""
    <hr style="border: none; height: 5px; background-color: #GGGGGG;" />
    """, unsafe_allow_html=True)

# Home tab content
if page == "Home":
   # Custom CSS for styling
    st.markdown("""
        <style>
        .overview {
            font-family: 'Arial', sans-serif;
            font-size: 34px;
            font-weight: bold;
            color: green;
        }
        .domain {
            font-family: 'Courier New', monospace;
            font-size: 30px;
            font-weight: bold;
            color: green;
        }
        .technologies {
            font-family: 'Verdana', sans-serif;
            font-size: 24px;
            font-weight: bold;
            color: lightblue;
        }
        .tech-list {
            font-family: 'Times New Roman', Times, serif;
            font-size: 20px;
            font-style: italic;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    # Applying the custom CSS classes
    st.markdown('<div class="overview">OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown('<div>The ultimate goal of this project is to develop a reliable predictive model that can pinpoint customers at high risk of loan default, allowing the financial institution to take proactive measures to mitigate potential losses and optimize their credit portfolio.</div>', unsafe_allow_html=True)
    st.markdown('<div class="domain">DOMAIN</div>', unsafe_allow_html=True)
    st.markdown('<div>Banking</div>', unsafe_allow_html=True)
    st.markdown('<div class="technologies">TECHNOLOGIES USED</div>', unsafe_allow_html=True)
    st.markdown("""
                <div class="tech-list">PYTHON</div>
                <div class="tech-list">DATA PREPROCESSING</div>
                <div class="tech-list">EDA</div>
                <div class="tech-list">PANDAS</div>
                <div class="tech-list">NUMPY</div>
                <div class="tech-list">VISUALIZATION</div>
                <div class="tech-list">MACHINE LEARNING</div>
                <div class="tech-list">STREAMLIT UI</div>
                """, unsafe_allow_html=True)

# Data Showcase tab content
elif page == "Data Showcase":
    st.header("Data Used")
    st.caption("Sample Data")
    st.dataframe(df.head())

    st.header("Model Performance")
    data = {
        "Algorithm": ["Decision Tree","Random Forest","KNN","XGradientBoost"],
        "Accuracy": [98,98,97,94],
        "Precision": [90,90,96,94],
        "Recall": [95,96,96,94],
        "F1 Score": [94,93,97,94]
    }
    dff = pd.DataFrame(data)
    st.dataframe(dff)
    st.markdown(f"## The Selected Algorithm is :orange[*KNN*] and its Accuracy is   :orange[*97%*]")


elif page == "ML Prediction":
    st.markdown(f'## :violet[*Predicting Customers Default on Loans*]')
    st.write('<h5 style="color:#FBCEB1;"><i>NOTE: Min & Max given for reference, you can enter any value</i></h5>', unsafe_allow_html=True)

    with st.form("my_form"):
        col1, col2 = st.columns([5, 5])
        
        with col1:
            TOTAL_INCOME = st.text_input("TOTAL INCOME (Min: 25650.0 & Max: 117000000.0)", key='TOTAL_INCOME')
            AMOUNT_CREDIT = st.text_input("CREDIT AMOUNT (Min: 45000.0 & Max: 4050000.0)", key='AMOUNT_CREDIT')
            AMOUNT_ANNUITY = st.text_input("ANNUITY AMOUNT (Min: 1980.0 & Max: 225000.0)", key='AMOUNT_ANNUITY')
            OCCUPATION_TYPE_CODE = st.selectbox("OCCUPATION TYPE", sorted(occupation_types.items()), format_func=lambda x: x[1], key='OCCUPATION_TYPE_CODE')[0] # type: ignore
            GENDER = st.selectbox("GENDER", list(gender_mapping.keys()), key='GENDER')
        with col2:
            OWN_CAR = st.selectbox("OWN CAR", list(own_car_mapping.keys()), key='OWN_CAR')
            EDUCATION_TYPE = st.selectbox("EDUCATION TYPE", list(education_type_mapping.keys()), key='EDUCATION_TYPE')
            HOUSING_TYPE = st.selectbox("HOUSING TYPE", list(housing_type_mapping.keys()), key='HOUSING_TYPE')
            FAMILY_STATUS = st.selectbox("FAMILY STATUS", list(family_status_mapping.keys()), key='FAMILY_STATUS')
            OBS_30_COUNT = st.text_input("OBS_30 COUNT (Min: 0 & Max: 348.0)", key='OBS_30_COUNT')
            DEF_30_COUNT = st.text_input("DEF_30 COUNT (Min: 0 & Max: 34.0)", key='DEF_30_COUNT')

        submit_button = st.form_submit_button(label="PREDICT STATUS")

    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #FBCEB1;
            color: purple;
            width: 50%;
            display: block;
            margin: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Validate input
    flag = 0 
    pattern = r"^(?:\d+|\d*\.\d+)$"

    for i in [TOTAL_INCOME, AMOUNT_CREDIT, AMOUNT_ANNUITY, OBS_30_COUNT, DEF_30_COUNT]:             
        if re.match(pattern, i):
            pass
        else:                    
            flag = 1  
            break

    if submit_button and flag == 1:
        if len(i) == 0:
            st.write("Please enter a valid number, space not allowed")
        else:
            st.write("You have entered an invalid value: ", i)  

    if submit_button and flag == 0:
        try:
            # Convert inputs to appropriate numeric types
            total_income = float(TOTAL_INCOME)
            amount_credit = float(AMOUNT_CREDIT)
            amount_annuity = float(AMOUNT_ANNUITY)
            occupation_type_code = int(OCCUPATION_TYPE_CODE)
            gender_code = gender_mapping[GENDER] # type: ignore
            own_car_code = own_car_mapping[OWN_CAR] # type: ignore
            education_type_code = education_type_mapping[EDUCATION_TYPE] # type: ignore
            housing_type_code = housing_type_mapping[HOUSING_TYPE] # type: ignore
            family_status_code = family_status_mapping[FAMILY_STATUS] # type: ignore
            obs_30_count = float(OBS_30_COUNT)
            def_30_count = float(DEF_30_COUNT)

            # Construct sample array for prediction
            sample = np.array([
                [
                    safe_sqrt(total_income),
                    safe_sqrt(amount_credit),
                    safe_sqrt(amount_annuity),
                    occupation_type_code,
                    gender_code,
                    own_car_code,
                    education_type_code,
                    housing_type_code,
                    family_status_code,
                    safe_sqrt(obs_30_count),
                    safe_sqrt(def_30_count)
                ]
            ])

            # Load the model
            with open(r"knnmodel.pkl", 'rb') as file:
                knn = pickle.load(file)
            
            # Perform prediction
          
            pred = knn.predict(sample)


            # Display prediction result
            if pred == 0:
                st.markdown(f' ## :grey[The status is :] :orange[Repay]')
            else:
                st.write(f' ## :orange[The status is ] :grey[Won\'t Repay]')

        except ValueError as e:
            st.error(f"Error processing inputs: {e}")

# ML Recommendation system tab content
elif page == "CHATBOT":
    run_chatbot()
    
# Data Visualization tab content
elif page == "Data Visualization":
    st.subheader("Insights of Bank Risk Controller System")
    
    # Assuming df1 is your DataFrame
    gender_counts = df['CODE_GENDER'].value_counts()
    family_status_counts = df['NAME_FAMILY_STATUS'].value_counts()

    # GENDER Bar Chart
    st.subheader("Gender Distribution")
    fig_gender = px.bar(gender_counts, x=gender_counts.index, y=gender_counts.values, labels={'x':'Gender', 'y':'Count'}, title="Gender Distribution")
    fig_gender.update_layout(yaxis_title="Count", xaxis_title="Gender", template="simple_white")
    st.plotly_chart(fig_gender, use_container_width=True)

    # FAMILY_STATUS Bar Chart
    st.subheader("Family Status Distribution")
    fig_family = px.bar(family_status_counts, x=family_status_counts.index, y=family_status_counts.values, labels={'x':'Family Status', 'y':'Count'}, title="Family Status Distribution")
    fig_family.update_layout(yaxis_title="Count", xaxis_title="Family Status", template="simple_white")
    st.plotly_chart(fig_family, use_container_width=True)
    
    
    occupation_type_counts = df['OCCUPATION_TYPE'].value_counts()
    
    # OCCUPATION_TYPE Bar CHart
    st.subheader("Occupation Type Distribution")
    fig_occupation = px.bar(occupation_type_counts, x=occupation_type_counts.index, y=occupation_type_counts.values, labels={'x':'Occupation Type', 'y':'Count'}, title="Occupation Type Distribution")
    fig_occupation.update_layout(yaxis_title="Count", xaxis_title="Occupation Type", template="plotly_dark")
    st.plotly_chart(fig_occupation, use_container_width=True)
    

    # Example: Line Chart for Income Type Counts
    income_type_counts = df['NAME_INCOME_TYPE'].value_counts()

    st.subheader("Income Type Distribution")
    fig_income = px.line(x=income_type_counts.index, y=income_type_counts.values, labels={'x':'Income Type', 'y':'Count'}, title="Income Type Distribution")
    fig_income.update_layout(yaxis_title="Count", xaxis_title="Income Type", template="seaborn")
    st.plotly_chart(fig_income, use_container_width=True)



        # Example: Pie Chart for Education Type
    education_type_counts = df['NAME_EDUCATION_TYPE'].value_counts()

    st.subheader("Education Type Distribution")
    fig_education = px.pie(education_type_counts, names=education_type_counts.index, values=education_type_counts.values, title="Education Type Distribution")
    fig_education.update_traces(textposition='inside', textinfo='percent+label')
    fig_education.update_layout(template="simple_white")
    st.plotly_chart(fig_education, use_container_width=True)

    
        # Example: Histogram for AMT_INCOME_TOTAL_sqrt
    st.subheader("AMT_INCOME_TOTAL Distribution")
    fig_income_hist = px.histogram(df, x='AMT_INCOME_TOTAL', nbins=30, title="AMT_INCOME_TOTAL Distribution")
    fig_income_hist.update_layout(yaxis_title="Frequency", xaxis_title="AMT_INCOME_TOTAL", template="simple_white")
    st.plotly_chart(fig_income_hist, use_container_width=True)


    fig = px.treemap(df, path=['OCCUPATION_TYPE', 'NAME_HOUSING_TYPE', 'NAME_FAMILY_STATUS'], 
                 title="Treemap of Occupation, Housing Type, and Family Status")
    st.plotly_chart(fig, use_container_width=True)


    #--------------------------------------------------------------4
   

    dff = df1[['AMT_INCOME_TOTAL_sqrt',
               'AMT_CREDIT_x_sqrt', 'AMT_ANNUITY_x_sqrt',
               'OCCUPATION_TYPE_sqrt', 'NAME_EDUCATION_TYPE_sqrt',
               'AMT_GOODS_PRICE_x_sqrt',
               'OBS_30_CNT_SOCIAL_CIRCLE_sqrt', "TARGET"]]

    # Calculate the correlation matrix
    corr = dff.corr().round(2)

    # Plot the heatmap using Plotly Express
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu",
                    title="Correlation Matrix Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# About tab content
elif page == "About":
    st.markdown("""
        ## About Bank Risk Controller System
        This application is developed as part of the Bank Risk Controller System project. It aims to provide a predictive model for identifying customers likely to default on their loans, leveraging machine learning and data analysis techniques.
    """)