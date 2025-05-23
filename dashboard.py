import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError

# Configuration
DATA_PATH = "employee_data.csv"
MODEL_PATH = "model.pkl"
FEATURE_NAMES = [
    'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education',
    'EducationField', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement',
    'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome',
    'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

# Load data with caching
@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        if 'Attrition' in df.columns and df['Attrition'].dtype == object:
            df['Attrition'] = df['Attrition'].map({'No': 0, 'Yes': 1})
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Load model with caching
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize app
st.set_page_config(page_title="HR Dashboard & Prediction", layout="wide", page_icon="📊")
df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

if df.empty or model is None:
    st.error("Failed to load data or model. Please check the files.")
    st.stop()

# Sidebar navigation
page = st.sidebar.radio("Menu", ["Dashboard Visualisasi", "Prediksi Karyawan"])

if page == "Dashboard Visualisasi":
    st.title("📊 Dashboard Analisis Karyawan")

    # Sidebar filter
    st.sidebar.header("Filter Data")

    # Pilih Department dengan selectbox agar tidak terlalu panjang
    department_filter = st.sidebar.multiselect(
        "Pilih Department",
        options=df['Department'].unique(),
        default=list(df['Department'].unique()),
        help="Filter data berdasarkan department karyawan"
    )

    gender_filter = st.sidebar.multiselect(
        "Pilih Gender",
        options=df['Gender'].unique(),
        default=list(df['Gender'].unique()),
        help="Filter data berdasarkan gender karyawan"
    )

    attrition_filter = st.sidebar.multiselect(
        "Pilih Status Attrition",
        options=['Stay', 'Leave'],
        default=['Stay', 'Leave'],
        help="Filter karyawan yang masih aktif (Stay) atau yang keluar (Leave)"
    )

    # Mapping filter attrition string ke numerik
    attrition_map = {'Stay': 0, 'Leave': 1}
    attrition_filter_num = [attrition_map[x] for x in attrition_filter]

    # Filter data
    df_filtered = df[
        (df['Department'].isin(department_filter)) &
        (df['Gender'].isin(gender_filter)) &
        (df['Attrition'].isin(attrition_filter_num))
    ]

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_employee = df_filtered['EmployeeId'].nunique()
        st.metric("👥 Total Karyawan", total_employee)

    with col2:
        attrition_rate = df_filtered['Attrition'].mean() * 100 if len(df_filtered) > 0 else 0
        st.metric("📉 Attrition Rate (%)", f"{attrition_rate:.2f}")

    with col3:
        avg_age = df_filtered['Age'].mean() if len(df_filtered) > 0 else 0
        st.metric("🎂 Usia Rata-rata", f"{avg_age:.1f} tahun")

    with col4:
        avg_years = df_filtered['YearsAtCompany'].mean() if len(df_filtered) > 0 else 0
        st.metric("⏳ Lama Bekerja Rata-rata", f"{avg_years:.1f} tahun")

    st.markdown("---")

    # Distribusi Usia & Gender
    age_gender_col1, age_gender_col2 = st.columns([3, 2])

    with age_gender_col1:
        fig_age = px.histogram(
            df_filtered, x='Age', nbins=20,
            title="Distribusi Usia Karyawan",
            labels={'Age': 'Usia (tahun)'},
            color='Gender',
            barmode='overlay',
            opacity=0.7
        )
        st.plotly_chart(fig_age, use_container_width=True)
        st.write("Histogram usia karyawan dengan warna berdasarkan gender.")

    with age_gender_col2:
        fig_gender = px.pie(
            df_filtered, names='Gender',
            title="Distribusi Gender",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_gender, use_container_width=True)
        st.write("Proporsi gender karyawan.")

    st.markdown("---")

    # Attrition Rate per Department dan Gender
    attr_dept_gender = df_filtered.groupby(['Department', 'Gender'])['Attrition'].mean().reset_index()

    fig_attr_dept_gender = px.bar(
        attr_dept_gender,
        x='Department', y='Attrition', color='Gender',
        barmode='group',
        labels={'Attrition': 'Attrition Rate'},
        title="Attrition Rate per Department dan Gender",
        text=attr_dept_gender['Attrition'].apply(lambda x: f"{x:.2%}")
    )
    fig_attr_dept_gender.update_traces(textposition='outside')
    fig_attr_dept_gender.update_layout(yaxis_tickformat='%')

    st.plotly_chart(fig_attr_dept_gender, use_container_width=True)

    st.markdown("---")

    # Hubungan YearsAtCompany dengan Attrition
    fig_years = px.box(
        df_filtered, x='Attrition', y='YearsAtCompany',
        title="Hubungan YearsAtCompany dengan Status Attrition",
        labels={'Attrition': 'Status Attrition (0=Stay,1=Leave)', 'YearsAtCompany': 'Tahun Bekerja'},
        color=df_filtered['Attrition'].map({0:'Stay', 1:'Leave'})  # warna kategorikal
    )
    st.plotly_chart(fig_years, use_container_width=True)
    st.write("Boxplot menunjukkan lama bekerja karyawan yang tetap dan yang keluar.")

    st.markdown("---")

    # Job Satisfaction berdasarkan Attrition dengan Bar Chart
    fig_job_satis = px.histogram(
        df_filtered,
        x='JobSatisfaction',
        color='Attrition',
        barmode='group',
        nbins=5,
        labels={'JobSatisfaction': 'Kepuasan Kerja', 'Attrition': 'Status Attrition'},
        title="Job Satisfaction berdasarkan Status Attrition"
    )
    st.plotly_chart(fig_job_satis, use_container_width=True)
    st.write("Distribusi kepuasan kerja berdasarkan status attrition.")

    st.markdown("---")

    # Heatmap korelasi numerik
    num_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    corr = df_filtered[num_cols].corr()

    fig_corr = px.imshow(
        corr, text_auto=True, aspect="auto",
        title="Heatmap Korelasi Antar Variabel Numerik"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.write("Dashboard ini dibuat menggunakan Streamlit dan Plotly untuk membantu analisis data karyawan dengan visualisasi interaktif dan filter dinamis.")
elif page == "Prediksi Karyawan":
    st.title("🤖 Prediksi Status Karyawan (Attrition)")
    
    # Create input form with columns
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Usia (Age)", min_value=18, max_value=70, value=30)
        business_travel = st.selectbox("Business Travel", options=df['BusinessTravel'].unique())
        department = st.selectbox("Department", options=df['Department'].unique())
        distance_from_home = st.number_input("Jarak dari Rumah ke Kantor (km)", min_value=1, max_value=100, value=10)
        education = st.selectbox("Education", options=df['Education'].unique())
        education_field = st.selectbox("Education Field", options=df['EducationField'].unique())
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
        gender = st.selectbox("Gender", options=df['Gender'].unique())
        
    with col2:
        hourly_rate = st.number_input("Hourly Rate", min_value=1, max_value=200, value=60)
        job_involvement = st.slider("Job Involvement", 1, 4, 3)
        job_level = st.selectbox("Job Level", options=df['JobLevel'].unique())
        job_role = st.selectbox("Job Role", options=df['JobRole'].unique())
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        marital_status = st.selectbox("Marital Status", options=df['MaritalStatus'].unique())
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
    
    # Additional inputs
    col3, col4 = st.columns(2)
    
    with col3:
        num_companies_worked = st.number_input("Jumlah Perusahaan Sebelumnya", min_value=0, max_value=20, value=2)
        over_time = st.selectbox("OverTime", options=df['OverTime'].unique())
        percent_salary_hike = st.number_input("Percent Salary Hike", min_value=0, max_value=100, value=15)
        performance_rating = st.slider("Performance Rating", 1, 4, 3)
        
    with col4:
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
        stock_option_level = st.selectbox("Stock Option Level", options=df['StockOptionLevel'].unique())
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=10)
        training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=2)
    
    col5, col6 = st.columns(2)
    
    with col5:
        work_life_balance = st.slider("Work Life Balance", 1, 4, 3)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        
    with col6:
        years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=3)
        years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=1)
        years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, max_value=20, value=2)

    # Create input dictionary
    input_data = {
        'Age': age,
        'BusinessTravel': business_travel,
        'Department': department,
        'DistanceFromHome': distance_from_home,
        'Education': education,
        'EducationField': education_field,
        'EnvironmentSatisfaction': environment_satisfaction,
        'Gender': gender,
        'HourlyRate': hourly_rate,
        'JobInvolvement': job_involvement,
        'JobLevel': job_level,
        'JobRole': job_role,
        'JobSatisfaction': job_satisfaction,
        'MaritalStatus': marital_status,
        'MonthlyIncome': monthly_income,
        'NumCompaniesWorked': num_companies_worked,
        'OverTime': over_time,
        'PercentSalaryHike': percent_salary_hike,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager
    }

    # Create DataFrame with only the expected features
    input_user = pd.DataFrame([{k: input_data[k] for k in FEATURE_NAMES if k in input_data}])

    # Preprocessing function
    @st.cache_resource
    def get_preprocessor():
        numeric_features = [
            'Age', 'DistanceFromHome', 'EnvironmentSatisfaction',
            'HourlyRate', 'JobInvolvement', 'JobSatisfaction',
            'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
            'PerformanceRating', 'RelationshipSatisfaction',
            'TotalWorkingYears', 'TrainingTimesLastYear',
            'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
            'YearsSinceLastPromotion', 'YearsWithCurrManager'
        ]
        
        categorical_features = [
            'BusinessTravel', 'Department', 'Education',
            'EducationField', 'Gender', 'JobLevel', 'JobRole',
            'MaritalStatus', 'OverTime', 'StockOptionLevel'
        ]
        
        # Only use features that exist in our FEATURE_NAMES
        numeric_features = [f for f in numeric_features if f in FEATURE_NAMES]
        categorical_features = [f for f in categorical_features if f in FEATURE_NAMES]
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor

    if st.button("Prediksi"):
        try:
            # Check for missing features
            missing_features = set(FEATURE_NAMES) - set(input_user.columns)
            if missing_features:
                st.error(f"Missing features: {missing_features}")
                st.stop()

            # Get and fit preprocessor
            preprocessor = get_preprocessor()
            X_train = df[FEATURE_NAMES]
            preprocessor.fit(X_train)
            
            # Transform input
            input_processed = preprocessor.transform(input_user)
            
            # Verify feature count
            if input_processed.shape[1] != model.n_features_in_:
                st.error(f"Feature mismatch! Model expects {model.n_features_in_} features, got {input_processed.shape[1]}")
                st.stop()
            
            # Make prediction
            prediction = model.predict(input_processed)[0]
            proba = model.predict_proba(input_processed)[0][1] if hasattr(model, "predict_proba") else None
            
            # Display results
            st.success(f"**Prediction:** {'Leave (Keluar)' if prediction == 1 else 'Stay (Bertahan)'}")
            
            if proba is not None:
                # Visualize probabilities
                prob_df = pd.DataFrame({
                    'Status': ['Stay', 'Leave'],
                    'Probability': [1-proba, proba]
                })
                
                fig = px.bar(
                    prob_df, 
                    x='Status', 
                    y='Probability',
                    color='Status',
                    color_discrete_map={'Stay': 'green', 'Leave': 'red'},
                    text='Probability',
                    title='Prediction Probabilities'
                )
                fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                fig.update_layout(yaxis_tickformat='.0%')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"**Probability of leaving:** {proba:.2%}")
                
                # Add risk assessment
                if proba > 0.7:
                    st.warning("🚨 High risk of attrition! Immediate action recommended.")
                elif proba > 0.5:
                    st.warning("⚠️ Moderate risk of attrition. Monitor closely.")
                else:
                    st.success("✅ Low risk of attrition.")
                    
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.error("Please ensure:")
            st.error("1. All input fields are filled correctly")
            st.error("2. The model matches the expected input features")
            st.error(f"Model expects {model.n_features_in_} features")