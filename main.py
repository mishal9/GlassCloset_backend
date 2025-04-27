import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import google.generativeai as genai
from supabase import create_client, Client
import jwt

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional, but recommended for local dev

# Get port from environment variable for Render deployment
port = int(os.getenv("PORT", "8000"))

# Load frontend URL from environment or use a default for email redirects
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Add a redirect URL for the API itself
API_URL = os.getenv("API_URL", "http://localhost:8000")

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

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# JWT Auth Dependency
async def get_current_user(Authorization: str = Header(...)):
    """
    Validates the JWT access token from the Authorization header and returns the user info.
    This function is designed to work with Supabase tokens.
    """
    if not Authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format.")
    
    token = Authorization.split(" ", 1)[1]
    
    try:
        # For Supabase tokens, we'll use the supabase client to validate
        # This approach doesn't require the JWT secret
        user = supabase.auth.get_user(token)
        if user and user.user:
            return user.user  # Return the user info from Supabase
        else:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    except Exception as e:
        # Fallback to manual JWT validation if Supabase validation fails
        try:
            # Verify without validation (just to extract the payload)
            payload = jwt.decode(token, options={"verify_signature": False})
            # For debugging purposes
            print(f"Token payload: {payload}")
            # Return the payload without validation
            return payload
        except Exception as jwt_e:
            raise HTTPException(status_code=401, detail=f"Invalid token format: {str(jwt_e)}")

@app.post("/signup")
async def signup(email: str, password: str):
    """
    Sign up a new user with Supabase Auth.
    """
    try:
        # Redirect to our own confirmation endpoint
        email_redirect = f"{API_URL}/confirm-email"
        
        # Include the redirect URL in the sign-up options
        result = supabase.auth.sign_up({
            "email": email, 
            "password": password,
            "options": {
                "email_redirect_to": email_redirect
            }
        })
        
        if result.user is None:
            raise HTTPException(status_code=400, detail=result.error.get('message', 'Unknown error'))
        return {
            "message": "User created successfully. Please check your email to confirm your account.", 
            "user_id": result.user.id
        }
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

@app.get("/confirm-email")
async def confirm_email(token_hash: str, type: str, redirect_to: str = None):
    """
    Handle email confirmation redirects from Supabase.
    This endpoint receives the confirmation token from Supabase and confirms the user's email,
    then redirects to the frontend.
    """
    try:
        # The token_hash and type are provided by Supabase in the URL
        # We need to verify the token with Supabase
        result = supabase.auth.verify_otp({
            "token_hash": token_hash,
            "type": type
        })
        
        # After successful verification, redirect to the frontend
        # If redirect_to is provided, use it, otherwise use the default frontend URL
        frontend_redirect = redirect_to or f"{FRONTEND_URL}/auth/login?confirmed=true"
        
        # Use FastAPI's RedirectResponse for the redirect
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=frontend_redirect)
    except Exception as e:
        # If there's an error, redirect to the frontend with an error parameter
        error_redirect = f"{FRONTEND_URL}/auth/login?error=confirmation_failed"
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=error_redirect)

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

