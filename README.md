# Intel Image Classification API

A FastAPI-based REST API for classifying images using a MobileNetV2 model trained on the Intel Image Classification dataset.

## Features

- Batch image classification with two processing modes
- Memory mode for small batches (fast)
- Disk mode for large batches (scalable)
- API key authentication
- Interactive API documentation

## Classes

The model predicts 6 categories: **buildings**, **forest**, **glacier**, **mountain**, **sea**, **street**

## Installation

```bash
git clone https://github.com/Zeyadelgabbas/Intel-Classifier.git
cd Intel-Classifier
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root:

```env
APP_NAME=Intel Image Classification API
VERSION=1.0.0
API_SECRET_KEY=your-secure-secret-key-here
```

## Usage

### Start Server

```bash
python main.py
```

Server runs at `http://127.0.0.1:8001`

### API Endpoints

**1. Home**
```
GET /
```

**2. Batch Classification (Memory)**
```
POST /Classify-batches-memory
```
Best for small batches (< 10 images)

**3. Batch Classification (Disk)**
```
POST /Classify-batches-disk
```
Best for large batches (> 10 images)

### Example: cURL

```bash
curl -X POST "http://127.0.0.1:8001/Classify-batches-memory" \
  -H "X-API-Key: your-api-key-here" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### Example: Python

```python
import requests

url = "http://127.0.0.1:8001/Classify-batches-memory"
headers = {"X-API-Key": "your-api-key-here"}

files = [
    ('files', open('mountain.jpg', 'rb')),
    ('files', open('sea.jpg', 'rb'))
]

response = requests.post(url, headers=headers, files=files)
print(response.json())
```

### Response Format

```json
{
  "predictions": [
    {
      "base_name": "image1.jpg",
      "class_index": 0,
      "class_name": "buildings",
      "confidence": 0.95
    }
  ]
}
```

## API Documentation

Interactive docs available at:
- Swagger UI: `http://127.0.0.1:8001/docs`
- ReDoc: `http://127.0.0.1:8001/redoc`

## Project Structure

```
Intel-Classifier/
├── Src/
│   ├── config.py          # Configuration
│   ├── inference.py       # Classifier
│   ├── schemas.py         # Response models
│   └── utils.py           # Utilities
├── artifacts/
│   ├── model.keras        # Trained model
│   └── idx2label.joblib   # Class mappings
├── main.py                # FastAPI app
├── requirements.txt
└── .env
```

## Model Info

- **Architecture**: MobileNetV2 (Transfer Learning)
- **Input Size**: 100x100 pixels
- **Classes**: 6
- **Preprocessing**: MobileNetV2 standard

## Troubleshooting

**Port already in use**
```bash
# Change port in main.py or kill existing process
lsof -i :8001  # Find process
kill -9 <PID>  # Kill it
```

**Wrong API key**
- Check `X-API-Key` header matches `.env` value

**Model not found**
- Ensure `artifacts/` folder contains `model.keras` and `idx2label.joblib`

## License

MIT License

---

**Author**: Zeyad Elgabbas  
**GitHub**: [Zeyadelgabbas](https://github.com/Zeyadelgabbas)