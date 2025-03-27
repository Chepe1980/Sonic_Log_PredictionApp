import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lasio
import io

# =============================================
# STREAMLIT APP CONFIGURATION
# =============================================
st.set_page_config(layout="wide")
st.title("DTSM Prediction from Well Logs 🛢️📊")
st.markdown("""
This app predicts **Shear Sonic (DTSM)** from well log data using a **Random Forest Regressor**.
Upload a CSV file, run the model, and export predictions.
""")

# Initialize a text box for processing comments
processing_log = st.empty()  # Placeholder for live updates

# =============================================
# 1. FILE UPLOAD SECTION
# =============================================
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=".csv")

if uploaded_file:
    processing_log.text("⏳ Loading CSV file...")
    df = pd.read_csv(uploaded_file)
    processing_log.text("✅ CSV file loaded successfully!")

    # Define features and target
    features = ['GR', 'RT', 'PHIE', 'NPHI', 'VCL', 'RHOBMOD', 'SW', 'DT']
    target = 'DTSM'
    
    # Clean data
    processing_log.text("🧹 Cleaning data (removing NaN values)...")
    df = df.dropna(subset=features + [target])
    processing_log.text(f"📊 Data cleaned! Shape: {df.shape}")

    # Show raw data preview
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

# =============================================
# 2. MODEL TRAINING & PREDICTION (ACTION BUTTON)
# =============================================
if uploaded_file and st.sidebar.button("Run Model"):
    processing_log.text("🚀 Starting model training...")
    
    # Train-test split
    processing_log.text("✂️ Splitting data into train/test sets...")
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    processing_log.text("🌳 Training Random Forest model (100 trees)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    processing_log.text("🎯 Model trained successfully!")
    
    # Predictions
    processing_log.text("🔮 Generating predictions...")
    df['DTSM_PRED'] = model.predict(X)
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.success("Model trained successfully!")
    st.write(f"**Mean Squared Error (MSE):** `{mse:.2f}`")
    st.write(f"**R² Score:** `{r2:.4f}`")
    
    # Create DTSM bins for visualization
    processing_log.text("📊 Creating DTSM bins for visualization...")
    df['DTSM_BINNED'] = pd.cut(df[target], bins=3, labels=['Low', 'Medium', 'High'])
    
    # =============================================
    # 3. VISUALIZATIONS (TABS)
    # =============================================
    processing_log.text("🎨 Generating visualizations...")
    st.subheader("Model Visualizations")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Pairplot (Feature Relationships)",
        "📈 Real vs Predicted DTSM",
        "📉 Depth Plot Comparison",
        "🔍 Feature Importance"
    ])
    
    with tab1:
        processing_log.text("🖌️ Rendering Pairplot...")
        
        # Create the pairplot with seaborn, and wrap it in a matplotlib figure for Streamlit
        fig1 = plt.figure(figsize=(10, 8))
        sns.pairplot(
            df,
            vars=features,
            hue='DTSM_BINNED',
            palette='viridis',
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 20},
            corner=True
        )
        
        # Display the plot in Streamlit
        st.pyplot(fig1)
        
        # Close the figure to prevent display issues with subsequent plots
        plt.close(fig1)
    
    with tab2:
        processing_log.text("📈 Rendering Real vs Predicted Scatter Plot...")
        
        # Ensure colors are selected from the same subset as y_test
        colors = df.loc[y_test.index, 'SW']  # This ensures colors match y_test's indices
        
        fig2 = plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        sc = plt.scatter(y_test, y_pred, c=colors, cmap='jet_r', alpha=0.6)
        
        # Add colorbar
        plt.colorbar(sc, label='SW (Water Saturation)')
        
        # Plot real vs predicted
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel('Real DTSM (us/ft)')
        plt.ylabel('Predicted DTSM (us/ft)')
        plt.title(f'Real vs Predicted DTSM (R² = {r2:.3f})')
        plt.grid(True)
        st.pyplot(fig2)
    
    with tab3:
        processing_log.text("📉 Rendering Depth Plot...")
        fig3 = plt.figure(figsize=(8, 12))
        plt.plot(df[target], df['DEPTH'], label='Real DTSM', color='blue', alpha=0.7)
        plt.plot(df['DTSM_PRED'], df['DEPTH'], label='Predicted DTSM', color='red', alpha=0.7, linestyle='--')
        plt.xlabel('DTSM (us/ft)')
        plt.ylabel('Depth (m)')
        plt.title('Real vs Predicted DTSM Log')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.grid(True)
        st.pyplot(fig3)
    
    with tab4:
        processing_log.text("🔍 Calculating Feature Importance...")
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig4 = plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        plt.title('Feature Importance for DTSM Prediction')
        plt.tight_layout()
        st.pyplot(fig4)
    
    processing_log.text("✅ All visualizations generated successfully!")

    # =============================================
    # 4. EXPORT PREDICTED CSV (ACTION BUTTON)
    # =============================================
    st.sidebar.subheader("Export Results")
    if st.sidebar.button("Export Predicted CSV"):
        processing_log.text("💾 Exporting predicted CSV file...")
        
        # Create CSV from the dataframe with the predictions
        csv = df.to_csv(index=False)
        st.sidebar.success(f"✅ File ready for download: `predicted_dtsm.csv`")
        
        # Provide download button for the CSV file
        st.download_button(
            label="Download Predicted CSV File",
            data=csv,
            file_name='predicted_dtsm.csv',
            mime='text/csv'
        )

# Clear processing log if no action is taken
if not uploaded_file:
    processing_log.text("ℹ️ Upload a CSV file to begin processing.")
