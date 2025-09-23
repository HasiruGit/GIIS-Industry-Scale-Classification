import streamlit as st

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from joblib import load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt

from lime import lime_tabular
from imblearn.over_sampling import SMOTE

## Load Model & data
df_imputed = pd.read_csv("C:/Users/Lenovo/OneDrive/Documents/Kuliah/KP/df_imputed.csv")


X = df_imputed.drop(columns="klasifikasi_encoded")
y = df_imputed.klasifikasi_encoded
#Identifikasi Numerik dan Kategorik
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(exclude='object').columns

#Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                                    

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#Model Training
rfmodel = RandomForestClassifier()
rfmodel.fit(X_train_resampled,y_train_resampled)
y_pred = rfmodel.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

label_mapping = {
    0: "Besar",
    1: "Menengah",
    2: "Kecil"
}

class_names = sorted(df_imputed["klasifikasi_encoded"].unique())
class_names_str = [label_mapping[i] for i in class_names]

# Save model and data
from joblib import dump, load

dump(rfmodel, "rfmodel.model")

rf_classif_2 = load("rfmodel.model")
y_test_preds = rf_classif_2.predict(X_test)

## Dashboard
st.title("Business Type :red[Prediction] :bar_chart: :chart_with_upwards_trend: :tea: :coffee:")
st.markdown("Prediksi Skala Industri dengan variabel-variabel pada GIIS")

tab1, tab2, tab3 = st.tabs(["Data", "Global Performance", "Local Performance and Prediction"])

with tab1:
    st.header("GIIS Dataset")
    st.write(df_imputed)

with tab2:
    st.header("Confusion Matrix | Feature Importances")
    col1, col2 = st.columns(2)
    with col1:
        conf_mat_fig = plt.figure(figsize=(6,6))
        ax1 = conf_mat_fig.add_subplot(111)
        ConfusionMatrixDisplay.from_predictions(y_test, y_test_preds, normalize='true', ax=ax1)
        st.pyplot(conf_mat_fig, use_container_width=True)

    with col2:
        feat_imp_fig = plt.figure(figsize=(6,6))
        ax1 = feat_imp_fig.add_subplot(111)
        skplt.estimators.plot_feature_importances(rf_classif_2, feature_names=df_imputed.columns.tolist(), ax=ax1, x_tick_rotation=90)
        st.pyplot(feat_imp_fig, use_container_width=True)

    st.divider()
    st.header("Classification Report")
    st.code(classification_report(y_test, y_test_preds))

with tab3:
    sliders = []
    feature_cols = ['nilai_inves', 'nilai_produksi', 'nilai_bb', 'total_tk', 'badan_usaha_encoded']
    col1, col2 = st.columns(2)
    with col1:
        for feature in feature_cols:
            ing_slider = st.slider(label=feature,min_value=float(df_imputed[feature].min()),max_value=float(min(df_imputed[feature].max(), 100000000)))
            sliders.append(ing_slider)

    with col2:
        col1, col2 = st.columns(2, gap="medium")
        
        prediction = rf_classif_2.predict([sliders])
        with col1:
            st.markdown("### Model Prediction : <strong style='color:tomato;'>{}</strong>".format(class_names_str[prediction[0]]), unsafe_allow_html=True)

        probs = rf_classif_2.predict_proba([sliders])
        probability = probs[0][prediction[0]]

        with col2:
            st.metric(label="Model Confidence", value="{:.2f} %".format(probability*100), delta="{:.2f} %".format((probability-0.5)*100))

        explainer = lime_tabular.LimeTabularExplainer(X_train.values, mode="classification", 
                                                      class_names=class_names_str, feature_names=df_imputed.columns.tolist())
        explanation = explainer.explain_instance(np.array(sliders), rf_classif_2.predict_proba, 
                                                 num_features=len(df_imputed.columns.tolist()), top_labels=3)
        interpretation_fig = explanation.as_pyplot_figure(label=prediction[0])
        st.pyplot(interpretation_fig, use_container_width=True)
    