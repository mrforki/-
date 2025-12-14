import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .orchestrator import get_reply_user # این خط تغییری نمی‌کند
# جایگزینی: sambanova به جای google.generativeai
from sambanova import SambaNova 

load_dotenv()

# --------------------------- مدل‌های ورودی ---------------------------
class UserMessage(BaseModel):
    user_message: str

class SummarizeRequest(BaseModel):
    text_to_summarize: str

# --------------------------- تنظیمات SambaNova ---------------------------
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
SAMBANOVA_BASE_URL = os.getenv("SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1")
summary_model_client = None
SETUP_ERROR = None
# مدل DeepSeek به صورت یک رشته تعریف می‌شود
MODEL_NAME = "DeepSeek-V3.1-Terminus" 

if SAMBANOVA_API_KEY:
    try:
        # مقداردهی به کلاینت SambaNova
        summary_model_client = SambaNova(
            api_key=SAMBANOVA_API_KEY,
            base_url=SAMBANOVA_BASE_URL,
        )
        print("✅ SambaNova API initialized successfully for summary.")
    except Exception as e:
        print("⚠️ خطا در مقداردهی SambaNova:", e)
        SETUP_ERROR = str(e)
else:
    SETUP_ERROR = "SAMBANOVA_API_KEY یافت نشد."
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
    # این تابع از orchestrator.py که در گام بعدی به روز رسانی می‌شود استفاده می‌کند
    return {"response": get_reply_user(data.user_message)}

# --------------------------- خلاصه‌سازی ---------------------------
@app.post("/summarize")
async def summarize_text(data: SummarizeRequest):
    global summary_model_client, SETUP_ERROR
    
    if not summary_model_client or SETUP_ERROR:
        raise HTTPException(status_code=500, detail=str(SETUP_ERROR))

    prompt = f"متن زیر را به فارسی و به صورت خلاصه توضیح بده:\n\n{data.text_to_summarize}"
    
    # ساختن لیست پیام‌ها برای خلاصه‌سازی
    messages = [
        {"role": "user", "content": prompt},
    ]

    try:
        # فراخوانی API چت SambaNova
        resp = summary_model_client.chat.completions.create(
            # اینجا نام مدل به صورت رشته (String) قرار گرفت
            model=MODEL_NAME, 
            messages=messages,
            temperature=0.1, # دمای پایین‌تر برای خلاصه‌سازی دقیق‌تر
        )
        
        if resp.choices and resp.choices[0].message:
            return {"summary": resp.choices[0].message.content.strip()}
        
        raise Exception("پاسخ نامعتبر از مدل دریافت شد.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در خلاصه‌سازی: {str(e)}")

# --------------------------- تست سلامت API ---------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "api_configured": SAMBANOVA_API_KEY is not None,
        # اینجا نام مدل به صورت رشته (String) قرار گرفت
        "model": MODEL_NAME 
    }