🔭 AI Vision Story Lab

An AI-powered multimodal vision application that analyzes images and generates captions, stories, and intelligent responses using Generative AI + Computer Vision.

🚀 Features
🔍 Image Analysis
Extracts structured insights like objects, mood, scene type, lighting, and colors.
✍️ Caption Generator
Generates captions in multiple styles:
Instagram
News
Poetic
Funny
📖 Story Generator
Creates AI-generated stories based on the uploaded image:
Adventure, Mystery, Sci-Fi, Romance, etc.
💬 Visual Q&A (VQA)
Ask questions about the image and get intelligent answers.
🎨 Color Palette Extraction
Detects dominant colors directly from the image.
🛠️ Tech Stack
Languages: Python
Frontend: Streamlit
AI/ML: OpenAI GPT-4o (Vision), LangChain
Libraries: Pillow (PIL), Tenacity
Concepts:
Generative AI
Multimodal AI
Computer Vision
Prompt Engineering
Visual Question Answering (VQA)
📂 Project Structure
AI-Vision-Story-Lab/
│── app.py            # Streamlit frontend
│── analyzer.py       # Core AI logic (analysis, story, prompts)
│── config.py         # Environment & config settings
│── .env              # API keys (not included in repo)
│── requirements.txt  # Dependencies
⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/AI-Vision-Story-Lab.git
cd AI-Vision-Story-Lab
2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Setup environment variables

Create a .env file:

OPENAI_API_KEY=your_api_key_here
VISION_MODEL=gpt-4o
▶️ Run the Application
streamlit run app.py
📸 Usage
Upload an image
Choose any feature:
Analyze
Generate captions
Create story
Ask questions
Get real-time AI responses
💡 Key Highlights
Built a multimodal AI system combining vision + language models
Implements end-to-end pipeline (image → processing → AI inference → output)
Uses structured JSON outputs for better interpretability
Designed for real-time interaction and scalability
🔮 Future Enhancements
User authentication system
Save & download generated content
Image history tracking
Deployment on cloud (AWS / Render / Railway)
Mobile-friendly UI
👨‍💻 Author

Kiran Kamble

AI/ML Engineer
Passionate about Generative AI & Real-world Applications
