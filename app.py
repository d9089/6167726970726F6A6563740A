import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
# @st.cache
def load_data():
    df = pd.read_csv("Crop_recommendation.csv")
    return df

df = load_data()


# Split dataset into training and testing sets
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models
models = {
    'Logistic Regression': LogisticRegression(random_state=2),
    'Decision Tree': DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=20, random_state=0)
}

# Define function to predict top crops
def predict_top_crops(model, inputs, top_n=3):
    probabilities = model.predict_proba(inputs)
    top_n_indices = (-probabilities).argsort()[:, :top_n]
    top_n_crops = [model.classes_[indices] for indices in top_n_indices]
    return top_n_crops

# Define Streamlit app
def main():
    st.title('Crop Recommendation System')

    # Sidebar title and selection
    st.sidebar.title('Crop Recommendation System')
    selected_classifier = st.sidebar.selectbox('Select Classifier', list(models.keys()))

    # Sidebar sliders for input parameters
    st.sidebar.subheader('Input Parameters')
    sn = st.sidebar.slider('NITROGEN (N)', 0.0, 150.0)
    sp = st.sidebar.slider('PHOSPHOROUS (P)', 0.0, 150.0)
    pk = st.sidebar.slider('POTASSIUM (K)', 0.0, 210.0)
    pt = st.sidebar.slider('TEMPERATURE', 0.0, 50.0)
    phu = st.sidebar.slider('HUMIDITY', 0.0, 100.0)
    pPh = st.sidebar.slider('pH', 0.0, 14.0)
    pr = st.sidebar.slider('RAINFALL', 0.0, 300.0)

    inputs = [[sn, sp, pk, pt, phu, pPh, pr]]

    # Predict top crops
    if selected_classifier:
        model = models[selected_classifier]
        model.fit(X_train, y_train)
        top_crops = predict_top_crops(model, inputs, top_n=3)
        st.subheader(f'Top 3 Predicted Crops using {selected_classifier}')
        for i, crops in enumerate(top_crops):
            st.write(f"{i+1}. {', '.join(crops)}")

    # Data exploration
    st.header('Data Exploration')

    # Display dataset
    st.write('### Dataset')
    st.write(df)
    # Histogram for prediction results of classifiers
    st.write('### Histogram for Prediction Results of Classifiers')

# Prepare data for histograms
    prediction_results = {}
    for classifier_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        prediction_results[classifier_name] = y_pred

# Create histograms
    for classifier_name, y_pred in prediction_results.items():
        st.subheader(f'Histogram for {classifier_name}')
        plt.figure(figsize=(10, 6))
        sns.histplot(y_pred, bins=len(set(y_pred)), kde=True, color='skyblue')
        plt.xlabel('Predicted Label')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Prediction Results for {classifier_name}')
        st.pyplot()


    # Confusion matrix
    if selected_classifier:
        st.write('### Confusion Matrix')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        model = models[selected_classifier]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        st.pyplot()
    #Box Plot    
    st.write('### Boxplot for Selected Feature')
    selected_feature = st.selectbox('Select Feature for Boxplot', X.columns)
    plt.figure(figsize=(10, 8))
    sns.boxplot(x='label', y=selected_feature, data=df, palette='Set3')
    plt.xlabel('Crop Label')
    plt.ylabel(selected_feature)
    plt.title(f'Boxplot of {selected_feature} by Crop Label')
    st.pyplot()
    #Histogram
    # Histogram for selected feature
    st.write('### Histogram for Selected Feature')
    selected_feature_hist = st.selectbox('Select Feature for Histogram', X.columns)
    plt.figure(figsize=(10, 8))
    sns.histplot(data=df, x=selected_feature_hist, kde=True, color='skyblue')
    plt.xlabel(selected_feature_hist)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {selected_feature_hist}')
    st.pyplot()


   
if __name__ == '__main__':
    main()