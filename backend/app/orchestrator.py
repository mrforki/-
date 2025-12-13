import os
from dotenv import load_dotenv
import httpx

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# مدل پیش‌فرض رایگان
DEFAULT_MODEL = "google/gemini-2.0-flash-exp:free"

SETUP_ERROR = None

if not OPENROUTER_API_KEY:
    print("⚠️ خطا: متغیر محیطی OPENROUTER_API_KEY تنظیم نشده است!")
    SETUP_ERROR = "کلید API (OPENROUTER_API_KEY) در فایل .env یافت نشد."
else:
    print("✅ OpenRouter API configured successfully")

def get_reply_user(user_text: str) -> str:
    """
    این تابع از OpenRouter API برای دریافت پاسخ استفاده می‌کند
    """
    if SETUP_ERROR:
        return f"⚠️ خطای تنظیمات: {SETUP_ERROR}"

    system_prompt = """تو یک دستیار هوشمند و دلسوز دانشجویان هستی که به زبان فارسی پاسخ می‌دهی.
وظیفه اصلی تو پاسخ دادن به سوالات درسی، پروژه‌ای، برنامه‌نویسی و ارائه راهنمایی‌های تحصیلی است.

قواعد پاسخگویی مهم:
1. همیشه پاسخی جامع، دقیق و متناسب با سطح دانشگاهی ارائه بده.
2. اگر کاربر در مورد «سازنده»، «توسعه‌دهنده» یا «چه کسی تو را ساخته» پرسید، بگو:
   «من توسط **محمدحسین تاجیک** با استفاده از OpenRouter AI توسعه داده شده‌ام. 
   هدف من کمک به رشد تحصیلی شماست. 
   می‌تونی محمدحسین رو در اینستاگرام دنبال کنی: https://www.instagram.com/mohmels/»
3. همیشه به زبان فارسی پاسخ بده مگر اینکه کاربر به زبان دیگری درخواست کند.
"""

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Student Chatbot"
                },
                json={
                    "model": DEFAULT_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            )

            if response.status_code == 200:
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    return "❌ پاسخی از سرور دریافت نشد."
                    
            elif response.status_code == 401:
                return "❌ کلید API نامعتبر است. لطفاً OPENROUTER_API_KEY را بررسی کنید."
                
            elif response.status_code == 429:
                return "⚠️ تعداد درخواست‌ها زیاد است. لطفاً کمی صبر کنید."
                
            else:
                return f"❌ خطای سرور OpenRouter (کد {response.status_code}): {response.text[:200]}"

    except httpx.TimeoutException:
        return "⏱️ درخواست زمان زیادی طول کشید. لطفاً دوباره تلاش کنید."
        
    except Exception as e:
        return f"❌ خطا در ارتباط با OpenRouter: {str(e)}"