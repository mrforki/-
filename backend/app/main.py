import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .orchestrator import get_reply_user
import google.generativeai as genai

load_dotenv()

# --------------------------- مدل‌های ورودی ---------------------------
class UserMessage(BaseModel):
    user_message: str

class SummarizeRequest(BaseModel):
    text_to_summarize: str

# --------------------------- تنظیمات Gemini ---------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
chat_model_client = None
SETUP_ERROR = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        chat_model_client = genai.GenerativeModel('gemini-1.5-flash')
        print("✅ Gemini API initialized successfully.")
    except Exception as e:
        print("⚠️ خطا در مقداردهی Gemini:", e)
        SETUP_ERROR = str(e)
else:
    SETUP_ERROR = "GEMINI_API_KEY یافت نشد."
    print("⚠️ ", SETUP_ERROR)

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
    global chat_model_client, SETUP_ERROR
    
    if not chat_model_client or SETUP_ERROR:
        raise HTTPException(status_code=500, detail=str(SETUP_ERROR))

    prompt = f"متن زیر را به فارسی و به صورت خلاصه توضیح بده:\n\n{data.text_to_summarize}"

    try:
        resp = chat_model_client.generate_content(prompt)
        return {"summary": resp.text.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در خلاصه‌سازی: {str(e)}")

# --------------------------- تست سلامت API ---------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "api_configured": GEMINI_API_KEY is not None,
        "model": "gemini-1.5-flash"
    }