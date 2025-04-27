import uvicorn
import os

if __name__ == "__main__":
    # Get port from environment variable (Render sets this)
    port = int(os.getenv("PORT", "8000"))
    
    # Start the server
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
