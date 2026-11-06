AI - Based Fabric Weave Reconstruction 

🧵 Fabric Weave AI

An intelligent web-based application designed to analyze fabric patterns using computer vision and image processing techniques. The system identifies fabric characteristics, detects repeat patterns, generates seamless fabric tiles, removes damaged regions, and produces detailed specification sheets in PDF format.

---

🚀 Live Demo

"Open Fabric Weave AI" (https://fabric-weave-ai-frontend.onrender.com)

Source Code: "GitHub Repository" (https://github.com/subi-tech382/fabric-weave-ai)

---

📌 Project Overview

Fabric Weave AI is designed to simplify fabric pattern analysis through automated image processing and computer vision.

Users can upload fabric images and obtain information about pattern structure, repeat dimensions, weave characteristics, colour distribution, and fabric specifications.

The application combines image processing algorithms with an interactive web interface to make fabric analysis easier and more accessible.

---

✨ Key Features

- Fabric Pattern Classification: Identifies fabric categories using the integrated classification model.
- Automatic Repeat Detection: Estimates repeating pattern dimensions using autocorrelation.
- Seamless Tile Generation: Generates repeatable fabric tiles.
- Damage Removal: Removes selected damaged regions using image inpainting.
- Weave Analysis: Estimates warp, weft, and yarn direction.
- Colour Palette Extraction: Extracts dominant colours and their percentage distribution.
- PDF Specification Sheet: Generates downloadable fabric analysis reports.
- PNG Export: Downloads processed fabric images and repeat tiles.

---

🛠️ Technology Stack

Component| Technologies
Frontend| HTML, CSS, JavaScript
Backend| Python, FastAPI
Image Processing| OpenCV
Numerical Computing| NumPy
PDF Generation| ReportLab
Deployment| Render
Version Control| GitHub

---

⚙️ System Architecture

User → Frontend Interface → FastAPI Backend → Image Processing Engine → Analysis Results → PNG / PDF Export

---

🔄 How It Works

1. Upload a fabric image.
2. The backend receives the image.
3. Image processing algorithms analyze the fabric.
4. Pattern and weave characteristics are estimated.
5. Seamless repeat tiles are generated.
6. The user can remove damaged regions.
7. Colour palette and fabric statistics are extracted.
8. A specification sheet is generated as a PDF.

---

🔌 API Endpoints

Method| Endpoint| Description
GET| "/"| Backend health check
POST| "/analyze"| Fabric image analysis
POST| "/spec-pdf"| PDF specification generation

API Documentation: "https://fabric-weave-ai.onrender.com/docs"

---

💻 Installation & Setup

Clone the repository:

git clone https://github.com/subi-tech382/fabric-weave-ai
cd fabric-weave-ai

Install backend dependencies:

pip install -r requirements.txt

Run the backend using the appropriate project entry point:

uvicorn app:app --reload

Open the frontend "index.html" or run it using a local development server.

Configure the frontend API URL to point to your backend.

---

🔮 Future Enhancements

- Improved fabric classification accuracy.
- Support for additional fabric categories.
- Advanced texture and pattern recognition.
- High-resolution fabric analysis.
- Enhanced reporting and analytics.

---

👩‍💻 Author

Subiksha R

B.Tech – Artificial Intelligence and Data Science

GitHub: https://github.com/subi-tech382

---

📄 License

This project is developed for educational and project demonstration purposes.
