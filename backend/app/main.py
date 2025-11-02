import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # برای اجازه دسترسی frontend
from .orchestrator import get_reply_user  # ایمپورت تابع از orchestrator

load_dotenv()

app = FastAPI()

# اضافه کردن CORS برای اینکه frontend بتونه به backend وصل شه (مثلاً از localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # برای تست، بعداً محدود کن
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/reply/{user_text}")
def reply(user_text: str):
    return {"reply": get_reply_user(user_text)}

# اگر نیاز به روت اصلی داری
@app.get("/")
def root():
    return {"message": "سلام! backend فعاله."}