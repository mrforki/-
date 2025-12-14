import os
from dotenv import load_dotenv
# جایگزینی: openai به جای sambanova
from openai import OpenAI 

load_dotenv()

# کلید API و URL گپ جی‌پی‌تی
GAPGPT_API_KEY = os.getenv("GAPGPT_API_KEY")
GAPGPT_BASE_URL = os.getenv("GAPGPT_BASE_URL", "https://api.gapgpt.app/v1") 

client = None
SETUP_ERROR = None
# استفاده از مدل قدرتمند gemini-2.5-pro
MODEL_NAME = 'gemini-2.5-pro' 

if not GAPGPT_API_KEY:
    print("⚠️ خطا: متغیر محیطی GAPGPT_API_KEY تنظیم نشده است!")
    SETUP_ERROR = "کلید API (GAPGPT_API_KEY) یافت نشد."
else:
    try:
        # مقداردهی به کلاینت OpenAI با URL و کلید گپ جی‌پی‌تی
        client = OpenAI(
            api_key=GAPGPT_API_KEY,
            base_url=GAPGPT_BASE_URL,
        )
        print("✅ GapGPT API initialized successfully.")
    except Exception as e:
        print(f"⚠️ خطا در تنظیم GapGPT API: {str(e)}")
        SETUP_ERROR = f"خطا در تنظیمات اولیه مدل: {str(e)}"

# System Instruction
SYSTEM_INSTRUCTION = """تو یک دستیار هوشمند و دلسوز دانشجویان هستی که به زبان فارسی پاسخ می‌دهی.
وظیفه اصلی تو پاسخ دادن به سوالات درسی، پروژه‌ای، برنامه‌نویسی و ارائه راهنمایی‌های تحصیلی است.

قواعد پاسخگویی مهم:
1. همیشه پاسخی جامع، دقیق و متناسب با سطح دانشگاهی ارائه بده.
2. اگر کاربر در مورد «سازنده»، «توسعه‌دهنده» یا «چه کسی تو را ساخته» پرسید، بگو:
   «من توسط **دانیال نادی** و **علی طاهری** با استفاده از تکنولوژی‌های پیشرفته هوش مصنوعی توسعه داده شده‌ام. 
   هدف من کمک به رشد تحصیلی و علمی دانشجویان عزیز است.»
3. همیشه به زبان فارسی پاسخ بده مگر اینکه کاربر به زبان دیگری درخواست کند.
"""

def get_reply_user(user_text: str) -> str:
    """
    این تابع از GapGPT API برای دریافت پاسخ چت استفاده می‌کند
    """
    global client, SETUP_ERROR
    
    if client is None:
        detailed_error = SETUP_ERROR if SETUP_ERROR else "خطای نامشخص در تنظیمات."
        return f"⚠️ خطای تنظیمات بک‌اند: {detailed_error}"

    try:
        # ساختن لیست پیام‌ها شامل System Instruction و پیام کاربر
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_text},
        ]
        
        # فراخوانی API چت (سازگار با OpenAI)
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=messages,
            temperature=0.7, 
            top_p=0.9
        )
        
        # استخراج پاسخ از شیء بازگشتی (ساختار OpenAI)
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()
        
        return "⚠️ پاسخ مناسبی از مدل دریافت نشد."
    
    except Exception as e:
        return f"❌ خطا در گرفتن پاسخ از GapGPT: {str(e)}"