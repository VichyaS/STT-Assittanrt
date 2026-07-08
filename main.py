# pip install sounddevice scipy openai gTTS playsound==1.2.2 requests

import os
import time
import base64
import requests
import sounddevice as sd
from scipy.io import wavfile
from openai import OpenAI
from gtts import gTTS
from playsound import playsound

# ==========================================
# 1. ตั้งค่าระบบและ API (ใช้ Environment Variable)
# ==========================================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    # Try loading from .env file
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    if key.strip() == "OPENROUTER_API_KEY":
                        OPENROUTER_API_KEY = val.strip().strip("\"'").strip()
                        break

if not OPENROUTER_API_KEY:
    print("❌ CRITICAL: OPENROUTER_API_KEY not set!")
    print("   Set the environment variable or create a .env file:")
    print("   OPENROUTER_API_KEY=sk-or-v1-your-key-here")
    print("   See .env.example for reference.")
    exit(1)

MODEL_NAME = "anthropic/claude-3-haiku"  # หรือใช้ "google/gemini-2.5-flash" เพื่อความเร็ว

# เรียกใช้ Client สำหรับ LLM Chat
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

SAMPLE_RATE = 44100
DURATION = 5  # ระยะเวลาอัดเสียง (วินาที)
AUDIO_INPUT_FILE = "user_command.wav"
AUDIO_OUTPUT_FILE = "vichya_response.mp3"

# ==========================================
# 2. ฟังก์ชันอัดเสียงจากไมโครโฟน
# ==========================================
def record_audio():
    print(f"\n🎙️ วิชญะ กำลังฟัง... (พูดได้เลย {DURATION} วินาที)")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # รอจนกว่าจะอัดเสียงเสร็จสมบูรณ์
    wavfile.write(AUDIO_INPUT_FILE, SAMPLE_RATE, recording)
    print("💾 บันทึกเสียงลงเครื่องเรียบร้อยแล้ว...")

# ==========================================
# 3. ฟังก์ชันแปลงเสียงเป็นข้อความ (STT) รองรับ OpenRouter
# ==========================================
def speech_to_text():
    print("🤖 กำลังแปลงเสียงเป็นข้อความ (ผ่าน Base64 JSON)...")
    
    # อ่านไฟล์เสียงแล้วเข้ารหัสเป็น Base64 string ตามข้อกำหนดของ OpenRouter
    with open(AUDIO_INPUT_FILE, "rb") as audio_file:
        audio_bytes = audio_file.read()
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
    
    url = "https://openrouter.ai/api/v1/audio/transcriptions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openai/whisper-large-v3",
        "input_audio": {
            "data": base64_audio,
            "format": "wav"
        },
        "language": "th"  # บังคับฟังและแกะภาษาไทย
    }
    
    # ส่งข้อมูลไปยัง OpenRouter Endpoint
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        result_json = response.json()
        return result_json.get("text", "")
    else:
        raise Exception(f"OpenRouter STT Error: {response.status_code} - {response.text}")

# ==========================================
# 4. ฟังก์ชันส่งข้อความให้ AI คิดหาคำตอบ (LLM Brain)
# ==========================================
def ask_vichya_brain(user_text):
    print(f"👤 คุณพูดว่า: '{user_text}'")
    print(f"🧠 กำลังส่งให้ {MODEL_NAME} ประมวลผลความคิด...")
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system", 
                "content": "คุณคือ วิชญะ ผู้ช่วยอัจฉริยะผู้หญิง  เมื่อได้รับคำสั่ง จงวิเคราะห์ปัญหาและคิดอย่างเป็นเหตุเป็นผลทีละขั้นตอน ตอบกลับสั้นกระชับ เป็นกันเอง เสียงสุภาพ และมีไหวพริบ โดยต้องตอบเป็นภาษาไทยเท่านั้น และ หากคำถามต้องการข้อมูลเชิงลึก ให้ใช้กระบวนการคิดแบบ ReAct"
            },
            {"role": "user", "content": user_text}
        ]
    )
    return response.choices[0].message.content

# ==========================================
# 5. ฟังก์ชันเปลี่ยนข้อความเป็นเสียงพูด (TTS) ด้วย playsound
# ==========================================
def text_to_speech(ai_response_text):
    print(f"🤖 วิชญะ คิดคำตอบ: '{ai_response_text}'")
    print("🔊 กำลังสังเคราะห์เสียงเพื่อตอบกลับ...")
    
    # 1. สร้างชื่อไฟล์ใหม่ทุกครั้งที่พูด เพื่อหนีปัญหา playsound ล็อกไฟล์เดิม
    timestamp = int(time.time())
    temp_filename = f"vichya_response_{timestamp}.mp3"
    
    try:
        # 2. แปลงข้อความเป็นเสียงแล้วเซฟด้วยชื่อไฟล์ใหม่
        tts = gTTS(text=ai_response_text, lang='th')
        tts.save(temp_filename)
        
        # 3. เล่นไฟล์เสียง
        playsound(temp_filename)
        
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการเล่นเสียง: {e}")
        
    finally:
        # 4. พยายามลบไฟล์เสียงทิ้งหลังจากพูดจบเพื่อไม่ให้รกเครื่อง 
        # (ใช้ try-except ครอบไว้ เพราะ playsound อาจจะยังล็อกไฟล์นี้อยู่ทำให้ลบไม่ได้ทันที)
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except:
            pass

# ==========================================
# 🏃♂️ ลูปการรันระบบหลัก (Main Pipeline)
# ==========================================
if __name__ == "__main__":
    # เพิ่ม while True เพื่อให้ระบบทำงานวนลูปไปเรื่อยๆ
    while True:
        try:
            # ขั้นตอนที่ 1: อัดเสียงของคุณผ่านไมโครโฟน
            record_audio()
            
            # ขั้นตอนที่ 2: แปลงเสียงเป็นข้อความ
            user_text = speech_to_text()
            
            if not user_text or user_text.strip() == "":
                print("❌ ระบบไม่ได้ยินเสียงพูด หรือไฟล์เสียงไม่มีข้อมูลตัวอักษร")
            
            else:
                # ขั้นตอนที่ 3: ส่งตัวอักษรให้สมองกล AI ประมวลผลคำตอบ
                ai_reply = ask_vichya_brain(user_text)
            
                # ขั้นตอนที่ 4: สังเคราะห์คำตอบของ AI ออกมาเป็นเสียงพูดทางลำโพง
                text_to_speech(ai_reply)
            
                print("⏳ จบรอบการทำงาน... เตรียมตัวฟังคำสั่งถัดไป\n")
                time.sleep(1) # หน่วงเวลา 1 วินาทีก่อนเริ่มฟังใหม่ (กันระบบรวน)

        # หากต้องการหยุดการวนลูป ให้กด Ctrl+C ในหน้าจอ Terminal
        except KeyboardInterrupt:
            print("\n🛑 ปิดระบบวิชญะเรียบร้อยแล้ว")
            break
            
        except Exception as e:
            print(f"\n⚠️ เกิดข้อผิดพลาดในระบบ: {e}")
            break