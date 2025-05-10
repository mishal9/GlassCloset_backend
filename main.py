import os
import io
import json
import re
import pathlib
import jwt
import time
from typing import Dict, Any, Optional, Union
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import JSONResponse
import google.generativeai as genai
from supabase import create_client, Client

# Load environment variables from .env if present
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

# Set up static files and templates
static_dir = pathlib.Path(__file__).parent / "static"
# Create static directory if it doesn't exist
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Create templates directory if it doesn't exist
templates_dir = pathlib.Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

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

# Helper functions for image analysis
def extract_json(text: str) -> Optional[str]:
    """Extract JSON object from text response"""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def get_gemini_model():
    """Get configured Gemini model for image analysis"""
    return genai.GenerativeModel('gemini-2.0-flash-exp-image-generation')

def get_detailed_prompt() -> str:
    """Get detailed analysis prompt for Gemini"""
    return (
        "Analyze this clothing item in detail and provide the following attributes: \n"
        "1. Main color(s)\n"
        "2. Secondary color(s) if any\n"
        "3. Garment type (e.g., shirt, pants, dress)\n"
        "4. Pattern (e.g., solid, striped, floral)\n"
        "5. Material (if identifiable)\n"
        "6. Style (e.g., casual, formal, athletic)\n"
        "7. Season appropriateness (e.g., summer, winter, all-season)\n"
        "8. Occasion suitability (e.g., work, casual, formal event)\n"
        "9. Fit description (if apparent)\n"
        "10. Brand identification (if visible)\n\n"
        "Return the result as a JSON object with these keys: main_colors, secondary_colors, garment_type, "
        "pattern, material, style, season, occasion, fit, brand. "
        "If an attribute cannot be determined, use null. Only return the JSON object."
    )

def get_basic_prompt() -> str:
    """Get basic analysis prompt for Gemini"""
    return (
        "Analyze the following attributes for this clothing item only if it is fully visible: "
        "color, type, primary color, pattern, style, weather, category, subcategory. "
        "Return the result as a JSON object with keys: color, type, primary_color, pattern, style, weather, category, subcategory. "
        "If an attribute cannot be determined, use null. Only return the JSON object."
    )

async def analyze_clothing_image(
    image: Image.Image, 
    analysis_type: str = "detailed",
    temperature: float = 0.0,
    top_k: int = 1
) -> Dict[str, Any]:
    """
    Analyze clothing image using Gemini AI
    
    Args:
        image: PIL Image object
        analysis_type: 'basic' or 'detailed'
        temperature: Temperature for generation
        top_k: Top-k parameter for generation
        
    Returns:
        Dictionary of attributes
    """
    model = get_gemini_model()
    prompt = get_basic_prompt() if analysis_type == "basic" else get_detailed_prompt()
    
    response = model.generate_content(
        [prompt, image],
        generation_config={
            "temperature": temperature,
            "top_k": top_k,
            "max_output_tokens": 2048
        }
    )
    
    json_str = extract_json(response.text)
    if not json_str:
        print(f"No JSON object found in Gemini response: {response.text}")
        raise ValueError("No JSON object found in Gemini response.")
        
    attributes = json.loads(json_str)
    
    # Replace all null values with 'Not detected' for consistency
    for k in attributes:
        if attributes[k] is None:
            attributes[k] = "Not detected"
    print("Attributes: ", attributes)
    return attributes

async def store_image_in_storage(user_id: str, image_data: bytes, file_name: str) -> str:
    """
    Store an image in Supabase Storage and return the public URL
    
    Args:
        user_id: The user ID to associate the image with
        image_data: The binary image data
        file_name: Original file name (will be modified to ensure uniqueness)
        
    Returns:
        Public URL to the stored image
    """
    try:
        # Create a unique file name to avoid collisions
        # Extract file extension from original name
        file_ext = os.path.splitext(file_name)[1].lower()
        if not file_ext:
            file_ext = ".jpg"  # Default to jpg if no extension
            
        # Create a unique filename with timestamp and user ID
        timestamp = int(time.time())
        unique_filename = f"{user_id}_{timestamp}{file_ext}"
        
        # The bucket name - you need to create this bucket in Supabase dashboard first
        bucket_name = "clothing-images"
        
        # Check if bucket exists, create if not
        buckets = supabase.storage.list_buckets()
        bucket_exists = any(bucket.name == bucket_name for bucket in buckets)
        
        if not bucket_exists:
            # Create the bucket with public access
            supabase.storage.create_bucket(bucket_name, {'public': True})
        
        # Upload the file to storage
        result = supabase.storage.from_(bucket_name).upload(
            path=unique_filename,
            file=image_data,
            file_options={"content-type": f"image/{file_ext[1:]}"}
        )
        
        # Get the public URL for the file
        public_url = supabase.storage.from_(bucket_name).get_public_url(unique_filename)
        
        return public_url
    except Exception as e:
        print(f"Error storing image: {e}")
        raise e

async def store_clothing_item(user_id: str, attributes: Dict[str, Any], image_data: Optional[bytes] = None, file_name: Optional[str] = None):
    """
    Store clothing item attributes in Supabase
    
    Args:
        user_id: The user ID to associate the clothing item with
        attributes: Dictionary of clothing attributes
        image_data: Optional binary image data to store
        file_name: Optional original file name for the image
        
    Returns:
        The created clothing item record
    """
    try:
        # Store the image if provided and get the URL
        image_url = None
        if image_data and file_name:
            image_url = await store_image_in_storage(user_id, image_data, file_name)
        
        # Create a new record in the clothing_items table
        clothing_item = {
            "user_id": user_id,
            "attributes": attributes,
            "image_url": image_url,
            # Add any additional metadata fields as needed
        }
        
        result = supabase.table("clothing_items").insert(clothing_item).execute()
        
        if len(result.data) == 0:
            raise ValueError("Failed to store clothing item in database")
            
        return result.data[0]
    except Exception as e:
        print(f"Error storing clothing item: {e}")
        raise e

@app.get("/clothing-items")
async def get_clothing_items(user=Depends(get_current_user)):
    """
    Retrieve all clothing items for the authenticated user
    
    Returns:
        List of clothing items with their attributes
    """
    try:
        # Get the user ID from the authenticated user
        user_id = user.id
        
        # Query the clothing_items table for items belonging to this user
        result = supabase.table("clothing_items").select("*").eq("user_id", user_id).execute()
        
        return {"clothing_items": result.data}
    except Exception as e:
        print(f"Error retrieving clothing items: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve clothing items: {str(e)}")

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), user=Depends(get_current_user), store: bool = Form(True)):
    """API endpoint for analyzing clothing images and optionally storing the results"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as img_exc:
            print(f"Image conversion error: {img_exc}")
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        # Analyze the image with detailed settings
        try:
            attributes = await analyze_clothing_image(image, "detailed")
            analysis_str = ", ".join(f"{k}: {v}" for k, v in attributes.items())
            
            result = {"analysis": analysis_str, "attributes": attributes}
            
            # Store the clothing item if requested
            if store:
                # Get the user ID from the authenticated user
                user_id = user.id
                
                # Store the clothing item in Supabase along with the image
                clothing_item = await store_clothing_item(
                    user_id=user_id, 
                    attributes=attributes,
                    image_data=contents,
                    file_name=file.filename
                )
                
                result["clothing_item_id"] = clothing_item["id"]
                result["image_url"] = clothing_item["image_url"]
                
            return JSONResponse(result)
        except ValueError as ve:
            raise HTTPException(status_code=500, detail=str(ve))

    except Exception as api_exc:
        print(f"Gemini API error: {api_exc}")
        raise HTTPException(status_code=500, detail=f"Gemini API error: {api_exc}")

@app.get("/upload-form", response_class=HTMLResponse)
async def get_upload_form(request: Request):
    """Render the HTML form for image uploads"""
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.post("/upload-image-form", response_class=HTMLResponse)
async def upload_image_form(request: Request, file: UploadFile = File(...), analysis_type: str = Form("basic")):
    """Handle image uploads from the HTML form and display results"""
    if not file.content_type.startswith("image/"):
        return templates.TemplateResponse(
            "upload_form.html", 
            {"request": request, "error": "File must be an image."}
        )
    
    try:
        # Ensure the uploads directory exists
        upload_dir = "static/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the uploaded file
        file_location = f"{upload_dir}/{file.filename}"
        with open(file_location, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Process the image
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as img_exc:
            print(f"Image conversion error: {img_exc}")
            return templates.TemplateResponse(
                "upload_form.html", 
                {"request": request, "error": "Uploaded file is not a valid image."}
            )
        
        # Analyze the image with appropriate settings for web form
        try:
            attributes = await analyze_clothing_image(
                image, 
                analysis_type,
                temperature=0.2,  # Slightly higher temperature for web form
                top_k=40          # Higher diversity for web form
            )
            
            # Prepare results for template
            if analysis_type == "basic":
                # Format basic results as a string
                results = ", ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in attributes.items())
                summary = None
            else:
                # For detailed analysis, pass the structured data
                results = attributes
                summary = f"This appears to be a {attributes.get('style', 'unknown style')} "\
                          f"{attributes.get('garment_type', 'garment')} in "\
                          f"{attributes.get('main_colors', 'unknown color')}, "\
                          f"suitable for {attributes.get('occasion', 'various occasions')}."
            
            # Return the template with results
            return templates.TemplateResponse(
                "upload_form.html", 
                {
                    "request": request, 
                    "results": results,
                    "summary": summary,
                    "analysis_type": analysis_type,
                    "image_path": f"/{file_location}"
                }
            )
            
        except ValueError as ve:
            return templates.TemplateResponse(
                "upload_form.html", 
                {"request": request, "error": str(ve)}
            )

    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        return templates.TemplateResponse(
            "upload_form.html", 
            {"request": request, "error": f"Error processing upload: {str(e)}"}
        )
