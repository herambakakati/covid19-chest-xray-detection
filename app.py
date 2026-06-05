import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gdown
import os

# =====================================================
# PAGE CONFIG (FIRST STREAMLIT COMMAND)
# =====================================================

st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# MODEL DOWNLOAD
# =====================================================

MODEL_PATH = "vgg16_weights.weights.h5"

if not os.path.exists(MODEL_PATH):

    url = "https://drive.google.com/uc?id=1eb-OCycqUVrRrM7BD39miq8Vbwef6DO9"

    with st.spinner("Downloading AI Model... Please wait."):
        gdown.download(
            url=url,
            output=MODEL_PATH,
            quiet=False
        )

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🩺",
    layout="wide"
)
# =====================================================
# PREMIUM CSS
# =====================================================

st.markdown("""
<style>

/* ==========================================
   GLOBAL
========================================== */

.stApp{
    background:
    linear-gradient(
        rgba(2,6,23,0.85),
        rgba(15,23,42,0.90)
    ),
    url("https://images.unsplash.com/photo-1576091160550-2173dba999ef?q=80&w=1920");

    background-size:cover;
    background-position:center;
    background-attachment:fixed;
}

.block-container{
    padding-top:1rem;
    max-width:1400px;
}

/* ==========================================
   SIDEBAR
========================================== */

section[data-testid="stSidebar"]{
    background:linear-gradient(
        180deg,
        #0f172a,
        #131c31,
        #1c2945
    );
    border-right:1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

.logo-circle{
    width:110px;
    height:110px;

    border-radius:50%;

    margin:auto;

    background:
    linear-gradient(
        135deg,
        #00f5ff,
        #4f46e5,
        #8b5cf6
    );

    display:flex;
    justify-content:center;
    align-items:center;

    font-size:55px;

    border:2px solid rgba(255,255,255,0.08);

    box-shadow:
        0 0 35px rgba(0,245,255,.7),
        0 0 70px rgba(79,70,229,.45),
        0 0 120px rgba(139,92,246,.25);

    animation:pulse 2.5s infinite;
}

@keyframes pulse{
    0%{transform:scale(1);}
    50%{transform:scale(1.08);}
    100%{transform:scale(1);}
}

.sidebar-title{
    text-align:center;
    font-size:28px;
    font-weight:800;
    margin-top:15px;
}

.sidebar-sub{
    text-align:center;
    color:#cbd5e1;
    margin-bottom:20px;
}

.live-status{
    text-align:center;
    color:#22c55e;
    font-weight:700;
    margin-bottom:20px;
}


/* ==========================================
   ULTRA PREMIUM SIDEBAR CARDS
========================================== */

.premium-card{
    position:relative;
    overflow:hidden;

    padding:22px;
    margin-bottom:20px;

    border-radius:28px;

    background:
    linear-gradient(
        145deg,
        rgba(24,35,65,0.95),
        rgba(15,23,42,0.98)
    );

    border:1px solid rgba(0,245,255,0.15);

    backdrop-filter:blur(25px);

    box-shadow:
        0 12px 35px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.05);

    transition:all .4s ease;
}

.premium-card::before{
    content:"";

    position:absolute;

    top:0;
    left:0;

    width:100%;
    height:3px;

    background:
    linear-gradient(
        90deg,
        #00f5ff,
        #4f46e5,
        #8b5cf6
    );
}

.premium-card::after{
    content:"";

    position:absolute;

    top:-60px;
    right:-60px;

    width:180px;
    height:180px;

    border-radius:50%;

    background:
    radial-gradient(
        rgba(0,245,255,0.15),
        transparent 70%
    );

    pointer-events:none;
}

.premium-card:hover{
    transform:translateY(-8px);

    border-color:rgba(0,245,255,0.35);

    box-shadow:
        0 20px 50px rgba(0,0,0,0.45),
        0 0 25px rgba(0,245,255,0.15);
}

.premium-card h4{
    color:white;
    font-size:22px;
    font-weight:800;
    margin-bottom:16px;
}

/* AI MODEL */

.model-badge{
    margin-top:15px;

    text-align:center;

    padding:16px;

    border-radius:18px;

    font-size:18px;
    font-weight:800;

    color:white;

    background:
    linear-gradient(
        135deg,
        rgba(0,245,255,0.15),
        rgba(79,70,229,0.25)
    );

    border:1px solid rgba(0,245,255,0.25);

    box-shadow:
        0 0 20px rgba(0,245,255,0.10);
}

/* TECHNOLOGY TAGS */

.tech-pill{
    display:inline-block;

    margin:5px;

    padding:8px 14px;

    border-radius:40px;

    font-size:12px;
    font-weight:700;

    color:white;

    background:
    linear-gradient(
        135deg,
        rgba(0,245,255,0.15),
        rgba(79,70,229,0.20)
    );

    border:1px solid rgba(0,245,255,0.15);

    transition:.3s;
}

.tech-pill:hover{
    transform:scale(1.05);

    border-color:#00f5ff;

    box-shadow:
        0 0 15px rgba(0,245,255,0.25);
}

/* ==========================================
   HERO
========================================== */

.hero-box{
    background:rgba(255,255,255,0.08);
    backdrop-filter:blur(30px);
    border-radius:35px;
    padding:30px 40px;
    margin-top:35px;
    border:1px solid rgba(0,245,255,.15);

    position:relative;
    overflow:hidden;

    box-shadow:
        0 15px 50px rgba(0,0,0,.35);

    margin-bottom:25px;

    transition:.4s;
}

.hero-box:hover{
    transform:translateY(-5px);

    border-color:rgba(0,245,255,.35);

    box-shadow:
        0 0 30px rgba(0,245,255,.25),
        0 0 60px rgba(139,92,246,.2);
}

.hero-tag{
    text-align:center;
    font-size:12px;
    letter-spacing:5px;
    color:#94a3b8;
    font-weight:700;
    margin-bottom:20px;
}

.hero-brand{
    display:flex;
    justify-content:center;
    align-items:center;
    gap:25px;
    margin-bottom:15px;
}

.hero-ai-icon{
    width:100px;
    height:100px;

    border-radius:28px;

    background:linear-gradient(
        135deg,
        #06b6d4,
        #2563eb,
        #7c3aed
    );

    display:flex;
    justify-content:center;
    align-items:center;

    font-size:55px;

    box-shadow:
        0 0 30px rgba(6,182,212,.5),
        0 0 60px rgba(124,58,237,.35);

    animation:floatAI 5s ease infinite;
}

.hero-title{
    font-size:64px;
    font-weight:900;
    line-height:1;

    background:linear-gradient(
        90deg,
        #ffffff,
        #00f5ff,
        #8b5cf6,
        #ffffff
    );

    background-size:300% auto;

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;

    animation:shine 6s linear infinite;
}

.hero-sub{
    text-align:center;
    font-size:22px;
    color:#cbd5e1;
    opacity:.95;
}

/* ==========================================
   PREDICTION CARD
========================================== */

.prediction-card{

    background:linear-gradient(
        135deg,
        rgba(0,255,120,.12),
        rgba(0,245,255,.08),
        rgba(99,102,241,.10)
    );

    backdrop-filter:blur(25px);

    border:1px solid rgba(0,255,120,.25);

    border-radius:30px;

    padding:35px;

    color:white;

    position:relative;

    overflow:hidden;

    box-shadow:
        0 0 30px rgba(0,255,120,.12),
        0 15px 40px rgba(0,0,0,.25);
}

.prediction-card::before{

    content:"";

    position:absolute;

    top:-50%;
    left:-50%;

    width:200%;
    height:200%;

    background:
    radial-gradient(
        circle,
        rgba(0,255,120,.08),
        transparent 60%
    );

    animation:rotateGlow 18s linear infinite;
}



            
/* ==========================================
   PREMIUM KNOWLEDGE CARDS
========================================== */

.knowledge-card{
    position:relative;
    overflow:hidden;

    border-radius:32px;
    padding:35px;

    min-height:420px;

    color:white;

    backdrop-filter:blur(25px);
    -webkit-backdrop-filter:blur(25px);

    transition:all 0.4s ease;

    box-shadow:
        0 15px 40px rgba(0,0,0,0.35);

    z-index:1;
}

.knowledge-card:hover{
    transform:translateY(-10px);

    box-shadow:
        0 25px 60px rgba(0,0,0,0.45),
        0 0 30px rgba(255,255,255,0.08);
}

.knowledge-card::before{
    content:"";

    position:absolute;
    inset:0;

    background:
    radial-gradient(
        circle at top right,
        rgba(255,255,255,0.12),
        transparent 55%
    );

    z-index:-1;
}

.covid-bg{
    background:
    linear-gradient(
        135deg,
        #081b44 0%,
        #173b8a 100%
    );

    border:2px solid #00f5ff;

    box-shadow:
        0 0 25px rgba(0,245,255,0.25);
}

.normal-bg{
    background:
    linear-gradient(
        135deg,
        #06361b 0%,
        #15803d 100%
    );

    border:2px solid #00ff88;

    box-shadow:
        0 0 25px rgba(0,255,136,0.20);
}

.viral-bg{
    background:
    linear-gradient(
        135deg,
        #5a2100 0%,
        #c2410c 100%
    );

    border:2px solid #f59e0b;

    box-shadow:
        0 0 25px rgba(245,158,11,0.20);
}

.knowledge-icon{
    text-align:center;

    font-size:72px;

    margin-bottom:18px;

    line-height:1;
}

.knowledge-title{
    text-align:center;

    font-size:34px;

    font-weight:900;

    color:white;

    margin-bottom:15px;

    letter-spacing:0.5px;
}

.knowledge-desc{
    text-align:center;

    color:#e2e8f0;

    font-size:16px;

    line-height:1.8;

    margin-bottom:25px;
}

.knowledge-list{
    color:white;

    font-size:18px;

    line-height:2;

    font-weight:500;

    padding-left:8px;
}

.knowledge-list br{
    margin-bottom:8px;
}

/* ==========================
   X-RAY IMAGE
========================== */

[data-testid="stImage"] img{

    border-radius:25px;

    border:1px solid rgba(255,255,255,.08);

    box-shadow:
        0 0 25px rgba(255,255,255,.08),
        0 15px 40px rgba(0,0,0,.35);
}

/* ==========================================
   FILE UPLOADER
========================================== */

[data-testid="stFileUploader"]{
    background:rgba(255,255,255,0.08);
    backdrop-filter:blur(20px);
    border-radius:24px;
    padding:15px;
    border:1px solid rgba(255,255,255,0.08);
}

[data-testid="stFileUploaderDropzone"]{
    background:rgba(255,255,255,0.03);
    border:2px dashed rgba(0,245,255,.25);
    border-radius:20px;
    padding:25px;
}

[data-testid="stFileUploader"] button{
    background:linear-gradient(
        135deg,
        #00f5ff,
        #4f46e5
    ) !important;

    color:white !important;

    border:none !important;

    border-radius:14px !important;
}

/* ==========================================
   PROGRESS BAR
========================================== */

.stProgress > div > div > div > div{

    background:
    linear-gradient(
        90deg,
        #00f5ff,
        #4f46e5,
        #00ff88
    );
}

/* ==========================================
   ANIMATIONS
========================================== */

@keyframes shine{
    0%{
        background-position:0%;
    }
    100%{
        background-position:300%;
    }
}

@keyframes floatAI{
    0%{
        transform:translateY(0);
    }
    50%{
        transform:translateY(-10px);
    }
    100%{
        transform:translateY(0);
    }
}
            
@keyframes rotateGlow{

    from{
        transform:rotate(0deg);
    }

    to{
        transform:rotate(360deg);
    }
}         

</style>
""", unsafe_allow_html=True)

# =====================================================
# SETTINGS
# =====================================================

CLASSES = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]

IMG_SIZE = 224

# =====================================================
# MODEL
# =====================================================

@st.cache_resource
def load_model():

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(3, activation="softmax")
    ])

    model.build((None,224,224,3))

    model.load_weights(
        "vgg16_weights.weights.h5"
    )

    return model

# =====================================================
# IMAGE PROCESSING
# =====================================================

def preprocess_image(image):

    image = np.array(image)

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = image.astype(np.float32)

    image = np.expand_dims(
        image,
        axis=0
    )

    image = preprocess_input(image)

    return image

# =====================================================
# LOAD MODEL
# =====================================================

try:
    model = load_model()

except Exception as e:
    st.error(f"Model Error: {e}")
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("""
    <div class="logo-circle">
    🩺
    </div>

    <div class="sidebar-title">
    AI Medical Assistant
    </div>

    <div class="sidebar-sub">
    Deep Learning Diagnosis Platform
    </div>

    <div class="live-status">
    ● LIVE SYSTEM ONLINE
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="premium-card">
    <h4>🎯 Detection Classes</h4>

    ✅ COVID-19<br>
    ✅ NORMAL<br>
    ✅ VIRAL PNEUMONIA
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="premium-card">
    <h4>🧠 AI Model</h4>

    <div class="model-badge">
    VGG16 Transfer Learning
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="premium-card">
    <h4>⚙ Technology</h4>

    <span class="tech-pill">TensorFlow</span>
    <span class="tech-pill">VGG16</span>
    <span class="tech-pill">OpenCV</span>
    <span class="tech-pill">Python</span>
    <span class="tech-pill">Streamlit</span>
    <span class="tech-pill">Deep Learning</span>

    </div>
    """, unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="hero-box">

<div class="hero-tag">
MEDICAL IMAGING INTELLIGENCE
</div>

<div class="hero-brand">

<div class="hero-ai-icon">
🧠
</div>

<div class="hero-title">
ChestVision AI
</div>

</div>
<div class="hero-sub">
Advanced Chest X-Ray Analysis Platform
</div>

</div>
""", unsafe_allow_html=True)

# =====================================================
# FILE UPLOAD
# =====================================================

st.markdown("""
<div style="
font-size:18px;
font-weight:700;
color:white;
margin-bottom:10px;
">
📤 Upload Chest X-Ray Image
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["jpg","jpeg","png"],
    label_visibility="collapsed"
)

# =====================================================
# PREDICTION
# =====================================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    processed_image = preprocess_image(
        image
    )

    with st.spinner(
        "🔬 AI analyzing chest X-Ray..."
    ):

        prediction = model.predict(
            processed_image,
            verbose=0
        )

    predicted_index = np.argmax(
        prediction
    )

    predicted_class = CLASSES[
        predicted_index
    ]

    confidence = (
        np.max(prediction) * 100
    )

    col1, col2 = st.columns([1.8,0.9])

    with col1:

        st.image(
            image,
            caption="Uploaded Chest X-Ray",
            use_container_width=True
        )

    with col2:

        st.markdown(
        f"""
        <div class="prediction-card">

        <div style="
        font-size:12px;
        letter-spacing:3px;
        color:#94a3b8;
        font-weight:700;
        margin-bottom:15px;
        ">
        AI DIAGNOSIS RESULT
        </div>

        <h1 style="
        font-size:60px;
        font-weight:900;
        margin-bottom:15px;
        color:white;
        ">
        {predicted_class}
        </h1>

        <div style="
        font-size:26px;
        font-weight:800;
        color:#00ff88;
        ">
        {confidence:.2f}% Confidence
        </div>

        </div>
        """,
        unsafe_allow_html=True
        )

        st.markdown("""
        <div style="
        font-size:40px;
        font-weight:800;
        color:white;
        margin-top:25px;
        margin-bottom:15px;
        ">
        Probability Distribution
        </div>
        """, unsafe_allow_html=True)

        for i, cls in enumerate(CLASSES):

            prob = float(
                prediction[0][i]
            )

            st.write(
                f"**{cls}: {prob*100:.2f}%**"
            )

            st.progress(prob)

# =====================================================
# INFORMATION
# =====================================================

st.markdown("""
<h2 style="
text-align:center;
color:white;
font-size:38px;
font-weight:900;
margin-bottom:30px;
">
🧠 AI Diagnostic Knowledge Base
</h2>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown("""
    <div class="knowledge-card covid-bg">
        <div class="knowledge-icon">🦠</div>
        <div class="knowledge-title">COVID-19</div>
        <div class="knowledge-desc">
            Coronavirus infection affecting lung tissues and respiratory pathways.
        </div>
        <div class="knowledge-list">
            ✓ Fever<br>
            ✓ Persistent Cough<br>
            ✓ Shortness of Breath<br>
            ✓ Chest Tightness
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown("""
    <div class="knowledge-card normal-bg">
        <div class="knowledge-icon">✅</div>
        <div class="knowledge-title">NORMAL</div>
        <div class="knowledge-desc">
            Healthy chest X-Ray with no significant abnormalities detected.
        </div>
        <div class="knowledge-list">
            ✓ Clear Lung Fields<br>
            ✓ Normal Anatomy<br>
            ✓ No Infiltration<br>
            ✓ Stable Condition
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown("""
    <div class="knowledge-card viral-bg">
        <div class="knowledge-icon">🫁</div>
        <div class="knowledge-title">Viral Pneumonia</div>
        <div class="knowledge-desc">
            Viral infection causing inflammation and opacity within lung tissues.
        </div>
        <div class="knowledge-list">
            ✓ Lung Inflammation<br>
            ✓ Respiratory Symptoms<br>
            ✓ Imaging Changes<br>
            ✓ Follow-up Recommended
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# PREMIUM FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <p style="
    text-align:center;
    color:#cbd5e1;
    font-size:14px;
    margin-bottom:4px;
    ">
    Powered by TensorFlow • VGG16 • Streamlit
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="
    text-align:center;
    color:#94a3b8;
    font-size:10px;
    letter-spacing:3px;
    margin-bottom:2px;
    ">
    DEVELOPED BY
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style="
    text-align:center;
    color:#00f5ff;
    margin-top:0px;
    ">
    HERAMBA KAKATI
    </h3>
    """,
    unsafe_allow_html=True
)
