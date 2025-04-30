# UKMLA Case Tutor API Deployment

This repository contains the backend API for the UKMLA Case Tutor application.

## Project Structure

```
UKMLA_API_Deployment_Files/
├── data/
│   └── cases/          # Case files directory
├── UKMLACaseBasedTutor7Cloud_FastAPI.py  # Main API code
├── requirements.txt    # Python dependencies
├── Procfile           # Deployment configuration
├── .env.example       # Environment variables template
└── .gitignore         # Git ignore rules
```

## Prerequisites

- Python 3.8+
- OpenAI API key
- Supabase account
- Railway/Render account

## Local Development Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your credentials
5. Run the API locally:
   ```bash
   uvicorn UKMLACaseBasedTutor7Cloud_FastAPI:app --reload
   ```

## Deployment

### Railway.app

1. Create a new project on Railway
2. Connect your GitHub repository
3. Add environment variables from your `.env` file
4. Deploy!

### Render.com

1. Create a new Web Service
2. Connect your GitHub repository
3. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn UKMLACaseBasedTutor7Cloud_FastAPI:app --host 0.0.0.0 --port $PORT`
4. Add environment variables
5. Deploy!

## API Endpoints

- `POST /start_case`: Start a new case
- `POST /continue_case`: Continue an existing case
- `GET /wards`: Get available wards and cases
- `POST /save_performance`: Save case performance data
- `GET /badges`: Get user badges
- `GET /progress`: Get user progress

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_ASSISTANT_ID`: Your OpenAI Assistant ID
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase API key

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License. 