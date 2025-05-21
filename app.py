import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plant_model.keras')

# Custom CSS for better styling
def load_css():
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #2E8B57, #90EE90);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .confidence-bar {
        background: #f0f0f0;
        border-radius: 10px;
        overflow: hidden;
        height: 25px;
        margin: 0.5rem 0;
    }

    
    .footer {
        text-align: center;
        color: #666;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

# Disease information dictionary
disease_info = {
    "Healthy": {
        "description": "The plant leaf appears to be healthy with no signs of disease.",
        "recommendations": [
            "Continue regular watering and monitoring",
            "Maintain proper sunlight exposure",
            "Keep the growing environment clean"
        ],
        "severity": "None",
        "color": "#4CAF50"
    },
    "Multiple Diseases": {
        "description": "The leaf shows signs of multiple diseases affecting the plant.",
        "recommendations": [
            "Isolate the plant to prevent spread",
            "Remove affected leaves immediately",
            "Consult with a plant pathologist",
            "Apply broad-spectrum fungicide if recommended"
        ],
        "severity": "High",
        "color": "#F44336"
    },
    "Rust": {
        "description": "Rust disease is a fungal infection that causes orange-brown spots on leaves.",
        "recommendations": [
            "Remove affected leaves and destroy them",
            "Improve air circulation around the plant",
            "Apply copper-based fungicide",
            "Avoid overhead watering"
        ],
        "severity": "Medium",
        "color": "#FF9800"
    },
    "Scab": {
        "description": "Scab is a fungal disease that causes dark, scaly lesions on leaves.",
        "recommendations": [
            "Prune affected areas",
            "Apply fungicide spray",
            "Ensure proper drainage",
            "Maintain good sanitation practices"
        ],
        "severity": "Medium",
        "color": "#2196F3"
    }
}

def create_confidence_chart(predictions, class_labels):
    """Create an interactive confidence chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=predictions,
            y=class_labels,
            orientation='h',
            marker=dict(
                color=predictions,
                colorscale='RdYlGn',
                cmin=0,
                cmax=1
            ),
            text=[f'{p:.1%}' for p in predictions],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence Levels",
        xaxis_title="Confidence",
        yaxis_title="Disease Types",
        height=300,
        showlegend=False,
        xaxis=dict(tickformat='.0%'),
        margin=dict(l=120, r=20, t=40, b=40)
    )
    
    return fig

def main():
    # Set page config
    st.set_page_config(
        page_title="Plant Disease Detector",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # Sidebar
    with st.sidebar:
        st.markdown("##  Instructions")
        st.markdown("""
        1. **Upload Image**: Choose a clear photo of a plant leaf
        2. **Wait for Analysis**: Our AI model will process the image
        3. **Review Results**: Check the prediction and confidence levels
        4. **Follow Recommendations**: Apply suggested treatments if needed
        """)
        
        st.markdown("##  Model Information")
        st.info("""
        **Dataset**: Plant Pathology 2020 - FGVC7
        
        **Classes Detected**:
        - 🟢 Healthy
        - 🔴 Multiple Diseases  
        - 🟠 Rust
        - 🔵 Scab
        """)
    
    # Main content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="main-header">🌱 Plant Disease Detector</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">AI-Powered Plant Health Analysis</div>', unsafe_allow_html=True)
    
    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📷 Upload Plant Leaf Image")
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            # Load model
            with st.spinner("Loading AI model..."):
                model = load_model()
            
            # Process image
            with st.spinner(" Analyzing image..."):
                # Read and process the image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Create two columns for image and results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### Uploaded Image")
                    st.image(image, caption="Original Image", use_container_width=True)
                
                # Preprocessing
                resized_image = cv2.resize(image, (512, 512))
                normalized_image = resized_image / 255.0
                input_image = np.expand_dims(normalized_image, axis=0)
                
                # Prediction
                predictions = model.predict(input_image)[0]
                class_labels = ["Healthy", "Multiple Diseases", "Rust", "Scab"]
                predicted_class_index = np.argmax(predictions)
                predicted_class = class_labels[predicted_class_index]
                confidence = predictions[predicted_class_index]
                
                with col2:
                    st.markdown("###  Analysis Results")
                    
                    # Main prediction
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"### Prediction: {predicted_class}")
                    st.markdown(f"### Confidence: {confidence:.1%}")
                    
                    # Confidence indicator
                    if confidence > 0.8:
                        st.success("High Confidence")
                    elif confidence > 0.6:
                        st.warning("Medium Confidence")
                    else:
                        st.error("Low Confidence")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed results section
            st.markdown("---")
            st.markdown("## Detailed Analysis")
            
            # Create three columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Prediction", predicted_class)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("## Confidence", f"{confidence:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                severity = disease_info[predicted_class]["severity"]
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Severity", severity)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Confidence chart
            st.markdown("###  All Class Probabilities")
            fig = create_confidence_chart(predictions, class_labels)
            st.plotly_chart(fig, use_container_width=True)
            
            # Disease information
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"###  About {predicted_class}")
            
            info = disease_info[predicted_class]
            st.markdown(f"**Description:** {info['description']}")
            
            st.markdown("**Recommended Actions:**")
            for i, rec in enumerate(info['recommendations'], 1):
                st.markdown(f"{i}. {rec}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional information
            if predicted_class != "Healthy":
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.markdown("Please try uploading a different image or check if the model file exists.")
    
    else:
        # Welcome message when no file is uploaded
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ## Welcome to Plant Disease Detector!
        
        This AI-powered tool helps you identify diseases in plant leaves quickly and accurately. 
        Simply upload a clear image of a plant leaf, and our trained model will:
        
        -  Analyze the leaf for signs of disease
        -  Provide confidence scores for each diagnosis
        -  Suggest treatment recommendations
        -  Show detailed analysis results
        
        **Get started by uploading an image above!**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()