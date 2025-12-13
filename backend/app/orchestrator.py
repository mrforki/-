import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
model = None
SETUP_ERROR = None

if not GEMINI_API_KEY:
    print("⚠️ خطا: متغیر محیطی GEMINI_API_KEY تنظیم نشده است!")
    SETUP_ERROR = "کلید API (GEMINI_API_KEY) یافت نشد."
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            system_instruction="""تو یک دستیار هوشمند و دلسوز دانشجویان هستی که به زبان فارسی پاسخ می‌دهی.
وظیفه اصلی تو پاسخ دادن به سوالات درسی، پروژه‌ای، برنامه‌نویسی و ارائه راهنمایی‌های تحصیلی است.

قواعد پاسخگویی مهم:
1. همیشه پاسخی جامع، دقیق و متناسب با سطح دانشگاهی ارائه بده.
2. اگر کاربر در مورد «سازنده»، «توسعه‌دهنده» یا «چه کسی تو را ساخته» پرسید، بگو:
   «من توسط **محمدحسین تاجیک** با استفاده از Google Gemini AI توسعه داده شده‌ام. 
   هدف من کمک به رشد تحصیلی شماست. 
   می‌تونی محمدحسین رو در اینستاگرام دنبال کنی: https://www.instagram.com/mohmels/»
3. همیشه به زبان فارسی پاسخ بده مگر اینکه کاربر به زبان دیگری درخواست کند.
"""
        )
        print("✅ Gemini API initialized successfully")
    except Exception as e:
        print(f"⚠️ خطا در تنظیم Gemini API: {str(e)}")
        SETUP_ERROR = f"خطا در تنظیمات اولیه مدل: {str(e)}"

def get_reply_user(user_text: str) -> str:
    """
    این تابع از Google Gemini API برای دریافت پاسخ استفاده می‌کند
    """
    if model is None:
        detailed_error = SETUP_ERROR if SETUP_ERROR else "خطای نامشخص در تنظیمات."
        return f"⚠️ خطای تنظیمات بک‌اند: {detailed_error}"

    try:
        response = model.generate_content(user_text)
        
        if response.candidates and response.candidates[0].finish_reason.name == 'SAFETY':
            return "⚠️ به دلیل خط‌مشی‌های ایمنی، امکان پاسخگویی به این سوال وجود ندارد."
        
        return response.text.strip()
    
    except Exception as e:
        return f"❌ خطا در گرفتن پاسخ از Gemini: {str(e)}"