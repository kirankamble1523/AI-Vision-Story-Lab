"""
app.py
──────
Streamlit frontend for the AI Vision Story Lab.

Run with:
    streamlit run app.py

Tabs:
  🔍 Analyze    → structured scene breakdown + color palette
  ✍️ Caption    → 4 caption styles (Instagram / News / Poetic / Funny)
  📖 Story      → 5 genre stories + custom prompt
  💬 Ask AI     → free-form Q&A chat about the image
"""

import os
import time

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

import vision_agent as va

# ─────────────────────────────────────────────────────────────────
load_dotenv()
# ─────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════
#  Page Config & Custom CSS
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Vision Story Lab",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #0f0f1a; color: #e8e8f0; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #2d2d5e;
    }

    /* Cards */
    .result-card {
        background: #1a1a2e;
        border: 1px solid #2d2d5e;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
        line-height: 1.75;
    }

    /* Color swatch */
    .swatch-row { display: flex; gap: 10px; margin-top: 10px; flex-wrap: wrap; }
    .swatch {
        width: 52px; height: 52px;
        border-radius: 8px;
        border: 2px solid rgba(255,255,255,0.1);
        display: flex; align-items: center; justify-content: center;
        font-size: 9px; color: rgba(0,0,0,0.6); font-weight: 600;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1a1a2e;
        border-radius: 8px 8px 0 0;
        border: 1px solid #2d2d5e;
        color: #9999cc;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #2d2d5e !important;
        color: #a78bfa !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6d28d9, #4f46e5);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Text areas and inputs */
    .stTextArea textarea, .stTextInput input {
        background: #1a1a2e !important;
        border: 1px solid #2d2d5e !important;
        color: #e8e8f0 !important;
        border-radius: 8px !important;
    }

    /* Story output */
    .story-output {
        background: #12122a;
        border-left: 4px solid #7c3aed;
        border-radius: 0 12px 12px 0;
        padding: 1.5rem 1.8rem;
        font-size: 1.02rem;
        line-height: 1.9;
        white-space: pre-wrap;
        margin-top: 1rem;
    }

    h1 { color: #a78bfa !important; }
    h2, h3 { color: #c4b5fd !important; }
    label { color: #c4b5fd !important; }

    /* Chat bubbles */
    .chat-user {
        background: #2d2d5e;
        border-radius: 12px 12px 4px 12px;
        padding: 0.8rem 1.1rem;
        margin: 0.5rem 0;
        text-align: right;
        color: #e8e8f0;
    }
    .chat-ai {
        background: #1a1a2e;
        border: 1px solid #2d2d5e;
        border-radius: 12px 12px 12px 4px;
        padding: 0.8rem 1.1rem;
        margin: 0.5rem 0;
        color: #e8e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════
#  Sidebar — Upload + API Status
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔭 Vision Story Lab")
    st.markdown("*Upload an image. Let AI see your world.*")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Supports JPG, PNG, WebP, BMP",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your image", use_column_width=True)
        w, h = image.size
        st.caption(f"📐 {w} × {h} px | {uploaded_file.type}")

    st.divider()

    # API key check
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and api_key.startswith("sk-"):
        st.success("✅ OpenAI API key loaded")
    else:
        st.error("❌ OPENAI_API_KEY missing")
        st.code("cp .env.example .env\n# then add your key")

    st.divider()
    st.markdown(
        "**Built with**\n"
        "- 🧠 GPT-4o Vision\n"
        "- 🦜 LangChain\n"
        "- 🖼️ Pillow\n"
        "- ⚡ Streamlit",
        unsafe_allow_html=False,
    )
    st.caption("College Project · AI Vision Story Lab")


# ═══════════════════════════════════════════════════════════════════
#  Main Area
# ═══════════════════════════════════════════════════════════════════

st.title("🔭 AI Vision Story Lab")
st.markdown(
    "Upload any image → **Analyze** it, generate **Captions**, "
    "write a **Story**, or **Ask** anything about it."
)

if not uploaded_file:
    # Landing state
    st.info("👈 Upload an image in the sidebar to get started.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 🔍 Analyze\nScene, objects, mood, colors, lighting")
    with col2:
        st.markdown("### ✍️ Caption\nInstagram · News · Poetic · Funny")
    with col3:
        st.markdown("### 📖 Story\nAdventure · Mystery · Sci-Fi · Romance")
    with col4:
        st.markdown("### 💬 Ask AI\nFree-form Q&A about your image")
    st.stop()


# Image is uploaded — show tabs
tab_analyze, tab_caption, tab_story, tab_qa = st.tabs(
    ["🔍 Analyze", "✍️ Caption", "📖 Story", "💬 Ask AI"]
)


# ═══════════════════════════════════════════════════════════════════
#  Tab 1: Analyze
# ═══════════════════════════════════════════════════════════════════

with tab_analyze:
    st.markdown("### 🔍 Scene Analysis")
    st.markdown(
        "GPT-4o examines your image and returns a structured breakdown of "
        "subjects, mood, lighting, colors, and more."
    )

    col_btn, col_space = st.columns([1, 3])
    with col_btn:
        run_analysis = st.button("▶ Analyze Image", key="btn_analyze", use_container_width=True)

    if run_analysis:
        with st.spinner("🧠 Analyzing your image..."):
            try:
                result = va.analyze_image(image)
                st.session_state["analysis"] = result
            except Exception as e:
                st.error(f"Error: {e}")

    if "analysis" in st.session_state:
        st.markdown(
            f'<div class="result-card">{st.session_state["analysis"]}</div>',
            unsafe_allow_html=True,
        )

    # Color Palette (runs without API)
    st.markdown("---")
    st.markdown("### 🎨 Dominant Color Palette")
    st.caption("Extracted directly from your image using Pillow (no API needed).")

    try:
        palette = va.extract_color_palette(image, num_colors=6)
        swatch_html = '<div class="swatch-row">'
        for r, g, b in palette:
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            # Pick black or white text based on luminance
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "#111" if luminance > 140 else "#eee"
            swatch_html += (
                f'<div class="swatch" style="background:{hex_color}; color:{text_color};">'
                f"{hex_color}</div>"
            )
        swatch_html += "</div>"
        st.markdown(swatch_html, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not extract palette: {e}")


# ═══════════════════════════════════════════════════════════════════
#  Tab 2: Caption
# ═══════════════════════════════════════════════════════════════════

with tab_caption:
    st.markdown("### ✍️ Caption Generator")
    st.markdown("Generate the perfect caption for any platform or purpose.")

    style_options = {
        "📸 Instagram": "instagram",
        "📰 News": "news",
        "🌿 Poetic": "poetic",
        "😂 Funny": "funny",
    }

    selected_styles = st.multiselect(
        "Choose caption style(s)",
        list(style_options.keys()),
        default=["📸 Instagram"],
    )

    if st.button("✍️ Generate Caption(s)", key="btn_caption", use_container_width=False):
        if not selected_styles:
            st.warning("Please select at least one style.")
        else:
            for label in selected_styles:
                style_key = style_options[label]
                with st.spinner(f"Writing {label} caption..."):
                    try:
                        caption = va.generate_caption(image, style=style_key)
                        st.markdown(f"**{label}**")
                        st.markdown(
                            f'<div class="result-card">{caption}</div>',
                            unsafe_allow_html=True,
                        )
                        # Copy button workaround — show text area for easy copy
                        with st.expander("📋 Copy text"):
                            st.text_area("", caption, height=100, key=f"copy_{style_key}")
                    except Exception as e:
                        st.error(f"{label} failed: {e}")


# ═══════════════════════════════════════════════════════════════════
#  Tab 3: Story
# ═══════════════════════════════════════════════════════════════════

with tab_story:
    st.markdown("### 📖 Story Generator")
    st.markdown(
        "GPT-4o looks at your image and writes a complete short story "
        "inspired by what it sees."
    )

    col_genre, col_custom = st.columns([1, 1])

    with col_genre:
        genre_options = {
            "⚔️ Adventure": "adventure",
            "🕵️ Mystery": "mystery",
            "💕 Romance": "romance",
            "🚀 Sci-Fi": "scifi",
            "🌈 Children's": "children",
        }
        selected_genre_label = st.selectbox(
            "Pick a genre",
            list(genre_options.keys()),
        )
        selected_genre = genre_options[selected_genre_label]

    with col_custom:
        custom_prompt = st.text_area(
            "Or write your own prompt (overrides genre)",
            placeholder=(
                "e.g. Write a horror story set in this location where "
                "something supernatural is about to happen..."
            ),
            height=120,
        )

    if st.button("📖 Generate Story", key="btn_story", use_container_width=False):
        with st.spinner(f"✍️ Writing your {selected_genre_label} story..."):
            try:
                story = va.generate_story(
                    image,
                    genre=selected_genre,
                    custom_prompt=custom_prompt or None,
                )
                st.session_state["story"] = story
                st.session_state["story_genre"] = selected_genre_label
            except Exception as e:
                st.error(f"Story generation failed: {e}")

    if "story" in st.session_state:
        st.markdown(f"**{st.session_state['story_genre']}**")
        st.markdown(
            f'<div class="story-output">{st.session_state["story"]}</div>',
            unsafe_allow_html=True,
        )
        word_count = len(st.session_state["story"].split())
        st.caption(f"📝 {word_count} words")

        with st.expander("📋 Copy story text"):
            st.text_area("", st.session_state["story"], height=200, key="story_copy")


# ═══════════════════════════════════════════════════════════════════
#  Tab 4: Ask AI (Q&A Chat)
# ═══════════════════════════════════════════════════════════════════

with tab_qa:
    st.markdown("### 💬 Ask Anything About Your Image")
    st.markdown(
        "Type any question — the AI looks at your image and answers. "
        "Try: *'What time of day is this?'* or *'How many people are in this image?'*"
    )

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Suggested questions
    st.markdown("**💡 Try these questions:**")
    suggestion_cols = st.columns(3)
    suggestions = [
        "What emotions does this image evoke?",
        "What season or time of year is this?",
        "What story is happening just outside the frame?",
        "What's the most interesting detail in this image?",
        "Is there anything unusual or out of place?",
        "How was this photo likely taken?",
    ]
    for i, suggestion in enumerate(suggestions):
        with suggestion_cols[i % 3]:
            if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                # Auto-fill the question
                st.session_state["prefill_question"] = suggestion

    st.divider()

    # Chat input
    prefill = st.session_state.pop("prefill_question", "")
    question = st.text_input(
        "Your question",
        value=prefill,
        placeholder="Ask anything about this image...",
        key="qa_input",
    )

    col_ask, col_clear = st.columns([1, 4])
    with col_ask:
        ask_btn = st.button("🔍 Ask", key="btn_ask", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear chat", key="btn_clear"):
            st.session_state.chat_history = []
            st.rerun()

    if ask_btn and question.strip():
        with st.spinner("🤔 Thinking..."):
            try:
                answer = va.ask_question(image, question)
                st.session_state.chat_history.append(
                    {"role": "user", "content": question}
                )
                st.session_state.chat_history.append(
                    {"role": "ai", "content": answer}
                )
            except Exception as e:
                st.error(f"Error: {e}")

    # Render chat history (newest first)
    if st.session_state.chat_history:
        st.markdown("#### Conversation")
        for msg in reversed(st.session_state.chat_history):
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">🧑 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-ai">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )