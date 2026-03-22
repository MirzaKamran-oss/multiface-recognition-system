# Multiface Recognition Application

A modern, production-ready multi-face recognition system built with FastAPI backend and React frontend (coming soon).

## 🚀 Features

- 🧠 **High Accuracy**: Uses DeepFace library for accurate face recognition
- 👥 **Multi-Face Recognition**: Recognizes multiple faces in a single image
- ⚡ **Fast & Async**: Built with FastAPI for high performance
- 🗄️ **Database Storage**: SQLite database for face embeddings
- 🔒 **RESTful API**: Well-structured API with comprehensive documentation
- 📊 **Health Monitoring**: Built-in health check endpoints
- 🎨 **Modern UI**: React frontend (in development)

## 📁 Project Structure

```
multiface-recognition-app/
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── api/         # API routes
│   │   ├── core/        # Configuration and database
│   │   ├── models/      # Database models
│   │   └── services/    # Business logic
│   ├── outputs/         # Recognized images output
│   ├── requirements.txt # Python dependencies
│   └── README.md       # Backend documentation
└── frontend/            # React frontend (coming soon)
```

## 🛠️ Tech Stack

### Backend
- **FastAPI**: Modern, fast web framework
- **DeepFace**: Face recognition library (no compilation required)
- **SQLAlchemy**: Database ORM
- **OpenCV**: Image processing
- **Python**: 3.8+ (compatible with multiple versions)

### Frontend (Coming Soon)
- **React**: Modern UI framework
- **TypeScript**: Type-safe development

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd multiface-recognition-app/backend
   ```

2. **Create virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   # Copy example environment file
   copy .env.example .env
   # or on Linux/Mac: cp .env.example .env
   ```
   
   Edit `.env` if needed (defaults work fine).

5. **Run the server:**
   ```bash
   uvicorn app.main:app --reload
   ```

   The API will be available at: `http://localhost:8000`

6. **Access API documentation:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## 📚 API Usage

### Train a Face

```bash
curl -X POST "http://localhost:8000/api/train/" \
  -F "person_id=1" \
  -F "name=John Doe" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### Recognize Faces

```bash
curl -X POST "http://localhost:8000/api/recognize/" \
  -F "file=@group_photo.jpg" \
  -o result.jpg
```

### List All Persons

```bash
curl http://localhost:8000/api/persons/
```

## 🔧 Configuration

Edit `backend/.env` to configure:

- `RECOGNITION_THRESHOLD`: Face recognition sensitivity (0.0-1.0, default: 0.4)
- `DETECTION_SIZE`: Face detection size (default: 640)
- `MAX_FILE_SIZE_MB`: Maximum upload size (default: 10MB)
- `DEBUG`: Enable debug mode (default: True)

## 🐛 Troubleshooting

### Dependencies Issues

If you encounter issues installing dependencies:

1. **Update pip:**
   ```bash
   pip install --upgrade pip
   ```

2. **Install dependencies one by one:**
   ```bash
   pip install fastapi uvicorn sqlalchemy
   pip install insightface onnxruntime opencv-python
   ```

### Face Recognition Not Working

- Ensure InsightFace models are downloaded (automatic on first use)
- Check that images contain clear, front-facing faces
- Verify `onnxruntime` is properly installed

### Port Already in Use

Change the port in `.env` or run:
```bash
uvicorn app.main:app --reload --port 8001
```

## 📝 Development

### Project Improvements Over Original

1. **Better Structure**: Organized into modules (api, core, models, services)
2. **Configuration Management**: Environment-based configuration with `.env`
3. **Error Handling**: Comprehensive error handling and validation
4. **API Documentation**: Auto-generated Swagger/ReDoc documentation
5. **Compatibility**: Works with Python 3.8+ (not just 3.11)
6. **Health Checks**: Built-in health monitoring
7. **CORS Support**: Ready for frontend integration

## 🎯 Next Steps

- [ ] React frontend implementation
- [ ] User authentication
- [ ] Real-time video feed support
- [ ] Cloud storage integration
- [ ] Analytics dashboard

## 📄 License

This project is open source and available for use.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
