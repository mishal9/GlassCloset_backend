import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import google.generativeai as genai
import tempfile
from supabase import create_client, Client
import jwt
from jwt import PyJWTError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional, but recommended for local dev

# Get port from environment variable for Render deployment
port = int(os.getenv("PORT", "8000"))

app = FastAPI(root_path=os.getenv("ROOT_PATH", ""))

# Allow CORS for local development (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Google Gemini API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")

genai.configure(api_key=GOOGLE_API_KEY)

# Load Supabase credentials from environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY environment variables not set.")
if not SUPABASE_JWT_SECRET:
    raise RuntimeError("SUPABASE_JWT_SECRET environment variable not set. Get it from your Supabase project settings.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# JWT Auth Dependency
async def get_current_user(Authorization: str = Header(...)):
    """
    Validates the JWT access token from the Authorization header and returns the user info.
    """
    if not Authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format.")
    token = Authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"])
        return payload  # Contains user info (e.g., sub, email)
    except PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {str(e)}")

@app.post("/signup")
async def signup(email: str, password: str):
    """
    Sign up a new user with Supabase Auth.
    """
    try:
        result = supabase.auth.sign_up({"email": email, "password": password})
        if result.user is None:
            raise HTTPException(status_code=400, detail=result.error.get('message', 'Unknown error'))
        return {"message": "User created successfully", "user_id": result.user.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Sign up failed: {str(e)}")

@app.post("/login")
async def login(email: str, password: str):
    """
    Log in a user with Supabase Auth.
    """
    try:
        result = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if result.user is None or result.session is None:
            raise HTTPException(status_code=401, detail=result.error.get('message', 'Invalid credentials'))
        return {"message": "Login successful", "access_token": result.session.access_token}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Login failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy", "message": "API is up and running"}

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), user=Depends(get_current_user)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    import io
    import json
    import re
    from PIL import Image

    try:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as img_exc:
            print(f"Image conversion error: {img_exc}")
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        # Use Gemini model for attribute extraction
        model = genai.GenerativeModel('gemini-2.0-flash-exp-image-generation')
        prompt = (
            "Analyze the following attributes for this clothing item only if it is fully visible: "
            "color, type, primary color, pattern, style, weather, category, subcategory. "
            "Return the result as a JSON object with keys: color, type, primary_color, pattern, style, weather, category, subcategory. "
            "If an attribute cannot be determined, use null. Only return the JSON object."
        )
        response = model.generate_content(
            [prompt, image],
            generation_config={
                "temperature": 0,
                "top_k": 1,
                "max_output_tokens": 2048
            }
        )
        def extract_json(text):
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return match.group(0)
            else:
                return None
        json_str = extract_json(response.text)
        if not json_str:
            print(f"No JSON object found in Gemini response: {response.text}")
            raise HTTPException(status_code=500, detail="No JSON object found in Gemini response.")
        try:
            attributes = json.loads(json_str)
        except Exception as parse_exc:
            print(f"Gemini response JSON parse error: {parse_exc}, raw JSON: {json_str}")
            raise HTTPException(status_code=500, detail=f"Failed to parse Gemini JSON: {parse_exc}")
        # Replace all null values with 'None' for consistency
        for k in attributes:
            if attributes[k] is None:
                attributes[k] = "None"
        analysis_str = ", ".join(f"{k}: {v}" for k, v in attributes.items())
        return JSONResponse({"analysis": analysis_str})
    except Exception as api_exc:
        print(f"Gemini API error: {api_exc}")
        raise HTTPException(status_code=500, detail=f"Gemini API error: {api_exc}")

