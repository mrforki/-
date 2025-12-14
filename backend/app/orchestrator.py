import os
from dotenv import load_dotenv
# جایگزینی: sambanova به جای google.generativeai
from sambanova import SambaNova 

load_dotenv()

# کلید API و URL پیش‌فرض SambaNova
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
SAMBANOVA_BASE_URL = os.getenv("SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1") 

client = None
SETUP_ERROR = None
# مدل DeepSeek برای چت
MODEL_NAME = 'DeepSeek-V3.1-Terminus' 

if not SAMBANOVA_API_KEY:
    print("⚠️ خطا: متغیر محیطی SAMBANOVA_API_KEY تنظیم نشده است!")
    SETUP_ERROR = "کلید API (SAMBANOVA_API_KEY) یافت نشد."
else:
    try:
        # مقداردهی به کلاینت SambaNova
        client = SambaNova(
            api_key=SAMBANOVA_API_KEY,
            base_url=SAMBANOVA_BASE_URL,
        )
        print("✅ SambaNova API initialized successfully.")
    except Exception as e:
        print(f"⚠️ خطا در تنظیم SambaNova API: {str(e)}")
        SETUP_ERROR = f"خطا در تنظیمات اولیه مدل: {str(e)}"

# System Instruction
SYSTEM_INSTRUCTION = """تو یک دستیار هوشمند و دلسوز دانشجویان هستی که به زبان فارسی پاسخ می‌دهی.
وظیفه اصلی تو پاسخ دادن به سوالات درسی، پروژه‌ای، برنامه‌نویسی و ارائه راهنمایی‌های تحصیلی است.

قواعد پاسخگویی مهم:
1. همیشه پاسخی جامع، دقیق و متناسب با سطح دانشگاهی ارائه بده.
2. اگر کاربر در مورد «سازنده»، «توسعه‌دهنده» یا «چه کسی تو را ساخته» پرسید، بگو:
   «من توسط **محمدحسین تاجیک** با استفاده از Google Gemini AI توسعه داده شده‌ام. 
   هدف من کمک به رشد تحصیلی شماست. 
   می‌تونی محمدحسین رو در اینستاگرام دنبال کنی: https://www.instagram.com/mohmels/»
3. همیشه به زبان فارسی پاسخ بده مگر اینکه کاربر به زبان دیگری درخواست کند.
"""

def get_reply_user(user_text: str) -> str:
    """
    این تابع از SambaNova API برای دریافت پاسخ چت استفاده می‌کند
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
        
        # فراخوانی API چت SambaNova
        response = client.chat.completions.create(
            model=MODEL_NAME, # استفاده از مدل DeepSeek
            messages=messages,
            temperature=0.7, 
            top_p=0.9
        )
        
        # استخراج پاسخ از شیء بازگشتی
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()
        
        return "⚠️ پاسخ مناسبی از مدل دریافت نشد."
    
    except Exception as e:
        # در SambaNova، خطاهای ایمنی یا فیلترینگ ممکن است به صورت استثنا (Exception) رخ دهند
        return f"❌ خطا در گرفتن پاسخ از SambaNova: {str(e)}"