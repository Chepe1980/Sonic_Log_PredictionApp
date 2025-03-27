import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lasio
import io  # For handling BytesIO
import time  # For simulating processing delays

# =============================================
# STREAMLIT APP CONFIGURATION
# =============================================
st.set_page_config(layout="wide")
st.title("DTSM Prediction from Well Logs 🛢️📊")
st.markdown("""
This app predicts **Shear Sonic (DTSM)** from well log data using a **Random Forest Regressor**.
Upload a LAS file, run the model, and export predictions.
""")

# Initialize a text box for processing comments
processing_log = st.empty()  # Placeholder for live updates

# =============================================
# 1. FILE UPLOAD SECTION
# =============================================
st.sidebar.header("Upload LAS File")
uploaded_file = st.sidebar.file_uploader("Choose a LAS file", type=".las")

if uploaded_file:
    try:
        processing_log.text("⏳ Loading LAS file...")
        
        # Read the file from the uploaded BytesIO object
        las = lasio.read(io.BytesIO(uploaded_file.read()))  # Wrap in BytesIO
        df = las.df().reset_index()
        
        processing_log.text("✅ LAS file loaded successfully!")

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

    except Exception as e:
        processing_log.text(f"❌ Error loading LAS file: {e}")
        st.error("Failed to load LAS file. Please check the file format.")

# =============================================
# 2. MODEL TRAINING & PREDICTION (ACTION BUTTON)
# =============================================
if uploaded_file and st.sidebar.button("Run Model"):
    try:
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
            st.pyplot(fig1)
        
        with tab2:
            processing_log.text("📈 Rendering Real vs Predicted Scatter Plot...")
            fig2 = plt.figure(figsize=(10, 8))
            colors = df.loc[y_test.index, 'SW']
            sc = plt.scatter(y_test, y_pred, c=colors, cmap='jet_r', alpha=0.6)
            plt.colorbar(sc, label='SW (Water Saturation)')
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
    
    except Exception as e:
        processing_log.text(f"❌ Error during model training or prediction: {e}")
        st.error("An error occurred during model training or prediction.")

# =============================================
# 4. EXPORT PREDICTED LAS (ACTION BUTTON)
# =============================================
st.sidebar.subheader("Export Results")
if st.sidebar.button("Export Predicted LAS"):
    try:
        processing_log.text("💾 Exporting predicted LAS file...")
        new_las = lasio.LASFile()
        
        # Copy header information
        for curve in las.curves:
            new_las.append_curve(
                curve.mnemonic,
                data=curve.data,
                unit=curve.unit,
                descr=curve.descr,
                value=curve.value
            )
        
        # Add curves
        new_las.append_curve('DEPTH', df['DEPTH'].values, unit='m')
        new_las.append_curve('DTSM', df[target].values, unit='us/ft', descr='Measured Shear Sonic')
        new_las.append_curve('DTSM_PRED', df['DTSM_PRED'].values, unit='us/ft', descr='Predicted Shear Sonic')
        
        # Save LAS file
        output_file = 'EJEM1_PREDICTED_DTSM.las'
        new_las.write(output_file)
        processing_log.text(f"✅ File exported as: `{output_file}`")
        st.sidebar.success(f"✅ File exported as: `{output_file}`")
    
    except Exception as e:
        processing_log.text(f"❌ Error during LAS export: {e}")
        st.error("An error occurred while exporting the LAS file.")

# Clear processing log if no action is taken
if not uploaded_file:
    processing_log.text("ℹ️ Upload a LAS file to begin processing.")
