import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from joblib import load, dump
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
from lime import lime_tabular
from imblearn.over_sampling import SMOTE

## Load Model & data
# Ensure the path is correct for your system
url = 'https://raw.githubusercontent.com/HasiruGit/GIIS-Industry-Scale-Classification/refs/heads/main/df_imputed.csv'
df_imputed = pd.read_csv(url)

# --- MODIFICATION ---
# Define the specific features you want to use for prediction AND training.
feature_cols = ['nilai_inves', 'nilai_produksi', 'nilai_bb', 'total_tk', 'badan_usaha_encoded']

X = df_imputed[feature_cols] # Use only the selected features
y = df_imputed.klasifikasi_encoded

#Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                                 
smote = SMOTE(random_state=42)
# We only resample the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#Model Training
rfmodel = RandomForestClassifier(random_state=42)
rfmodel.fit(X_train_resampled, y_train_resampled)
y_pred = rfmodel.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}") # This will print to your console

# Save and load model
dump(rfmodel, "rfmodel.model")
rf_classif_2 = load("rfmodel.model")
y_test_preds = rf_classif_2.predict(X_test)

# Class mapping
label_mapping = {
    0: "Besar",
    1: "Menengah",
    2: "Kecil"
}
# Get class names from the original 'y' series to ensure correct order
class_names = sorted(y.unique()) 
class_names_str = [label_mapping[i] for i in class_names]

## Dashboard
st.title("Business Type :red[Prediction] :bar_chart: :chart_with_upwards_trend: :tea: :coffee:")
st.markdown("Prediksi Skala Industri dengan variabel-variabel pada GIIS")

tab_home, tab_data, tab_global, tab_local = st.tabs([
    "Home", 
    "Data", 
    "Global Performance", 
    "Local Performance and Prediction"
])

# --- UPDATED "Home" TAB ---
with tab_home:
  # Create columns for the images at the top left
    img_col1, img_col2, img_col3, img_col4, _ = st.columns([1, 1, 1, 1, 6], gap='medium') 
    img1 = "https://raw.githubusercontent.com/HasiruGit/GIIS-Industry-Scale-Classification/main/logo.png"
    img2 = "https://raw.githubusercontent.com/HasiruGit/GIIS-Industry-Scale-Classification/main/Lambang-ITS-2-300x300.png"
    img3 = 'https://raw.githubusercontent.com/HasiruGit/GIIS-Industry-Scale-Classification/main/Statistika%20ITS.png'
  
    with img_col1:
        st.image(img1, caption="Disperidag Jawa Timur", width=100) 
    with img_col2:
        st.image(img2, caption="ITS", width=100)
    with img_col3:
        st.image(img3, caption="Statistika ITS", width=100)
      
    st.header("Disperindag Jawa Timur")
    
    # 1. Deskripsi Instansi
    st.subheader("Dinas Perindustrian dan Perdagangan Provinsi Jawa Timur")
    # Added a unique key
    st.markdown("Dinas Perindustrian dan Perdagangan Provinsi (Disperindag) Jawa Timur memiliki " \
    "visi untuk menjadikan Jawa Timur maju dengan mengedepankan semangat pro investasi melalui " \
    "langkah yang tertata dan penuh totalitas (METAL). Misi yang diemban adalah melanjutkan pembangunan " \
    "berbasis investasi yang maju, sinergis, dan berkelanjutan; menghadirkan pemerintahan yang bersih, efektif, " \
    "dan terpercaya; serta mewujudkan masyarakat Jawa Timur yang tangguh, cerdas, berkarakter, dan berdaya saing "
    "(Dinas Perindustrian dan Perdagangan Provinsi Jawa Timur, n.d.)")
    
    # 2. Latar belakang
    st.subheader("Deskripsi Proyek")
    # Added a unique key
    st.markdown("""- Tujuan umum dari pelaksanaan Kerja Praktik ini adalah mendapatkan gambaran umum tentang Dinas Perindustrian dan Perdagangan Provinsi Jawa Timur secara lengkap, meliputi profil, sejarah, visi misi, struktur organisasi, serta penerapan bidang kerja statistik dalam mendukung perumusan suatu kebijakan. Selain itu, kegiatan ini bertujuan agar mahasiswa mampu beradaptasi dan bermasyarakat dalam dunia kerja serta mendapatkan pengalaman nyata tentang lingkungan kerja di Dinas Perindustrian dan Perdagangan Provinsi Jawa Timur. 
    - Tujuan khusus dari pelaksanaan Kerja Praktik ini adalah untuk menganalisis permasalahan yang diberikan oleh Dinas Perindustrian dan Perdagangan Provinsi Jawa Timur dan dapat menyajikan solusi yang tepat dan dapat digunakan oleh instansi terhadap permasalahan yang ditemukan atau sesuai dengan kebutuhan instansi. """)
    

    # 3. Expandable Summary Statistics (with new plots)
    st.divider()
    with st.expander("Show Data Summary Statistics"):
        st.subheader("Descriptive Statistics Table")
        st.dataframe(df_imputed.describe())
        
        st.subheader("Data Visualizations")
        
        # --- PLOT 1: Target Variable Distribution ---
        st.markdown("##### Target Variable Distribution (Original Data)")
        target_counts = df_imputed['klasifikasi_encoded'].map(label_mapping).value_counts()
        
        fig_target = plt.figure(figsize=(7, 5))
        target_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
        plt.title("Distribution of Business Scale (Target)")
        plt.ylabel("Count")
        plt.xlabel("Scale")
        plt.xticks(rotation=0)
        st.pyplot(fig_target)

        # --- PLOT 2: Missing Data Distribution ---
        st.markdown("##### Missing Data Distribution")
        missing_pct = df_imputed.isnull().sum() / len(df_imputed) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        
        if missing_pct.empty:
            st.success("No missing data found in the dataset. ✨")
        else:
            fig_missing = plt.figure(figsize=(10, 6))
            missing_pct.plot(kind='bar', color='salmon')
            plt.title("Percentage of Missing Data by Column")
            plt.ylabel("Percentage Missing (%)")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_missing)

        # --- PLOT 3: Numerical Column Distributions ---
        st.markdown("##### Numerical Column Distributions")
        st.markdown("_Shows bar charts for discrete data (e.g., encoded IDs) and log-histograms for continuous data (e.g., financial values)._")
        
        numerical_cols = df_imputed.select_dtypes(include=np.number).columns
        numerical_cols = numerical_cols.drop('klasifikasi_encoded', errors='ignore') 
        
        for col in numerical_cols:
            fig_num = plt.figure(figsize=(8, 4))
            ax = fig_num.add_subplot(111)
            unique_vals = df_imputed[col].nunique()
            
            if unique_vals < 30: # Treat as discrete
                counts = df_imputed[col].value_counts().sort_index()
                counts.plot(kind='bar', ax=ax, color='cyan', edgecolor='black')
                ax.set_title(f"Distribution of {col} (Discrete)")
                ax.set_xlabel("Value")
                plt.xticks(rotation=0)
            else: # Treat as continuous
                try:
                    log_data = np.log1p(df_imputed[col].dropna())
                    ax.hist(log_data, bins=30, color='lightcoral', edgecolor='black')
                    ax.set_title(f"Distribution of Log({col}) (Continuous)")
                    ax.set_xlabel(f"Log({col})")
                except Exception as e:
                    ax.hist(df_imputed[col].dropna(), bins=30, color='lightcoral', edgecolor='black')
                    ax.set_title(f"Distribution of {col} (Continuous)")
                    ax.set_xlabel(col)
            
            ax.set_ylabel("Frequency")
            st.pyplot(fig_num, clear_figure=True)

        # --- PLOT 4: Categorical Column Distributions ---
        st.markdown("##### Categorical Column Distributions")
        categorical_cols = df_imputed.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            st.info("No categorical/object columns found in the dataset.")
        
        for col in categorical_cols:
            fig_cat = plt.figure(figsize=(10, 5))
            ax = fig_cat.add_subplot(111)
            
            counts = df_imputed[col].value_counts()
            
            if len(counts) > 30:
                counts = counts.head(30)
                ax.set_title(f"Top 30 Categories for {col}")
            else:
                ax.set_title(f"Distribution of {col}")
                
            counts.plot(kind='bar', ax=ax, color='lightgreen', edgecolor='black')
            ax.set_ylabel("Count")
            ax.set_xlabel("Category")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig_cat, clear_figure=True)

    # --- NEW SECTION ADDED ---
    st.divider() # Add a separator
    st.subheader("Kesimpulan")
    
    # Add a text area with a unique key
    st.markdown("Data masih banyak yang inbalance, banyak missing data yang diganti menjadi 0 merupakan salah" \
    "satu alasan. Hal ini disebabkan dikarenakan pengisian data yang kurang teratur. Pemilik industri cenderung kurang paham" \
    "dengan struktur dan urgensi pengisian data. maka dari itu dilakukan pemodelan klasifikasi dengan harapan " \
    "dapat memudahkan proses pengisian data. Namun tidak dapat dipungkiri masalah imbalance data, sehingga " \
    "dilakukan metode SMOTE untuk mengatasi masalah tersebut.")

# --- (Rest of the code remains unchanged) ---

with tab_data:
    st.header("GIIS Dataset")
    st.write(df_imputed)

with tab_global:
    st.header("Confusion Matrix | Feature Importances")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        conf_mat_fig = plt.figure(figsize=(6,6))
        ax1 = conf_mat_fig.add_subplot(111)
        ConfusionMatrixDisplay.from_predictions(y_test, y_test_preds, 
                                                normalize='true', 
                                                ax=ax1, 
                                                display_labels=class_names_str)
        st.pyplot(conf_mat_fig)

    with col2:
        st.subheader("Feature Importances")
        feat_imp_fig = plt.figure(figsize=(6,6))
        ax1 = feat_imp_fig.add_subplot(111)
        skplt.estimators.plot_feature_importances(rf_classif_2, 
                                                  feature_names=feature_cols, 
                                                  ax=ax1, 
                                                  x_tick_rotation=90)
        st.pyplot(feat_imp_fig)

    st.divider()
    st.header("Classification Report")
    st.code(classification_report(y_test, y_test_preds, target_names=class_names_str))

with tab_local:
    st.header("Make a Prediction")
    sliders = []
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Features")
        for feature in feature_cols:
            max_val = float(df_imputed[feature].max())
            if max_val > 1e9: 
                max_val = 1e9 
                
            ing_slider = st.slider(label=feature, 
                                   min_value=float(df_imputed[feature].min()), 
                                   max_value=max_val,
                                   value=float(df_imputed[feature].mean())
                                  )
            sliders.append(ing_slider)

    with col2:
        st.subheader("Prediction & Explanation")
        col1_pred, col2_pred = st.columns(2, gap="medium")
        
        prediction_input = [sliders]
        prediction = rf_classif_2.predict(prediction_input)
        
        with col1_pred:
            st.markdown("### Model Prediction : <strong style='color:tomato;'>{}</strong>"
                        .format(class_names_str[prediction[0]]), 
                        unsafe_allow_html=True)

        probs = rf_classif_2.predict_proba(prediction_input)
        probability = probs[0][prediction[0]]

        with col2_pred:
            st.metric(label="Model Confidence", 
                      value="{:.2f} %".format(probability * 100))
        
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            mode="classification",
            class_names=class_names_str,
            feature_names=feature_cols
        )
        
        explanation = explainer.explain_instance(
            data_row=np.array(sliders), 
            predict_fn=rf_classif_2.predict_proba, 
            num_features=len(feature_cols),
            top_labels=3
        )
        
        interpretation_fig = explanation.as_pyplot_figure(label=prediction[0])
        st.pyplot(interpretation_fig)







