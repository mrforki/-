import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .orchestrator import get_reply_user
import httpx

load_dotenv()

# --------------------------- مدل‌های ورودی ---------------------------
class UserMessage(BaseModel):
    user_message: str

class SummarizeRequest(BaseModel):
    text_to_summarize: str

# --------------------------- تنظیمات OpenRouter ---------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# مدل‌های رایگان پیشنهادی
FREE_MODELS = {
    "gemini": "google/gemini-2.0-flash-exp:free",
    "llama": "meta-llama/llama-3.2-3b-instruct:free",
    "mistral": "mistralai/mistral-7b-instruct:free"
}

SETUP_ERROR = None

if not OPENROUTER_API_KEY:
    SETUP_ERROR = "کلید API (OPENROUTER_API_KEY) در فایل .env یافت نشد."
    print(f"⚠️ {SETUP_ERROR}")
else:
    print("✅ OpenRouter API Key loaded successfully.")

# --------------------------- اپ اصلی ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static_files", StaticFiles(directory="frontend"), name="frontend_static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Frontend file not found.")

# --------------------------- چت ---------------------------
@app.post("/reply")
async def reply(data: UserMessage):
    return {"response": get_reply_user(data.user_message)}

# --------------------------- خلاصه‌سازی ---------------------------
@app.post("/summarize")
async def summarize_text(data: SummarizeRequest):
    global SETUP_ERROR
    
    if SETUP_ERROR:
        raise HTTPException(status_code=500, detail=str(SETUP_ERROR))

    prompt = f"متن زیر را به فارسی و به صورت خلاصه توضیح بده:\n\n{data.text_to_summarize}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": os.getenv("APP_URL", "https://student-chatbot.up.railway.app"),
                    "X-Title": "Student Chatbot"
                },
                json={
                    "model": FREE_MODELS["gemini"],
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.5,
                    "max_tokens": 1000
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"OpenRouter API Error: {response.text[:200]}"
                )
            
            result = response.json()
            summary = result["choices"][0]["message"]["content"]
            
            return {"summary": summary.strip()}

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="درخواست به OpenRouter زمان زیادی طول کشید")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در خلاصه‌سازی: {str(e)}")

# --------------------------- تست سلامت API ---------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "api_configured": OPENROUTER_API_KEY is not None,
        "available_models": list(FREE_MODELS.keys()),
        "default_model": FREE_MODELS["gemini"]
    }