import os
import tempfile
import re
from typing import Optional
import subprocess
import imageio_ffmpeg
import whisper
import whisper.audio  


FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
print("Using FFmpeg binary:", FFMPEG_BIN)
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_BIN)
os.environ["FFMPEG_BINARY"] = FFMPEG_BIN

# Patch whisper.audio.run to use exact ffmpeg
_original_run = whisper.audio.run
def run_ffmpeg_patch(cmd, **kwargs):
    cmd[0] = FFMPEG_BIN
    return _original_run(cmd, **kwargs)
whisper.audio.run = run_ffmpeg_patch

# Test FFmpeg
try:
    subprocess.run([FFMPEG_BIN, "-version"], check=True)
    print("FFmpeg test passed.")
except Exception as e:
    print("FFmpeg test failed:", e)


WHISPER_MODEL_NAME = "base"
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)


KNOWN_LOCATIONS = [
    "kitchen", "living room", "bedroom", "bathroom", "garage", "office",
    "dining room", "kids room", "study", "storeroom", "balcony", "hall",
    "laundry", "pantry"
]

DEVICE_KEYWORDS = [
    "fridge", "refrigerator", "ac", "air conditioner", "washing machine", 
    "microwave", "oven", "heater", "dishwasher", "fan", "tv", "television",
    "cooler", "freezer", "pump", "dryer"
]

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}


from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

class ParseRequest(BaseModel):
    text: str

class DeviceDetails(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    powerwatts: Optional[int] = None
    rating: Optional[int] = None
    location: Optional[str] = None


def _to_int_safe(value) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(round(float(value)))
    except Exception:
        return None

def _extract_power_watts(text: str) -> Optional[int]:
    t = text.lower()
    power_pat = re.compile(
        r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>k(?:ilo)?w(?:att)?s?|w(?:att)?s?)\b",
        flags=re.I
    )
    for m in power_pat.finditer(t):
        num = float(m.group("num"))
        unit = m.group("unit").lower()
        if unit.startswith("kw") or unit.startswith("kilo"):
            return _to_int_safe(num * 1000.0)
        else:
            return _to_int_safe(num)
    m2 = re.search(r"(?:power|wattage)\s*[:=]?\s*(\d+)", t, flags=re.I)
    if m2:
        return _to_int_safe(m2.group(1))
    return None

def _extract_rating(text: str) -> Optional[int]:
    t = text.lower()
    m = re.search(r"\b(\d+)\s*[-\s]*star[s]?\b", t)
    if m:
        return _to_int_safe(m.group(1))
    m2 = re.search(r"\b({})\s*[-\s]*star[s]?\b".format("|".join(NUMBER_WORDS.keys())), t)
    if m2:
        return NUMBER_WORDS.get(m2.group(1))
    m3 = re.search(r"\brating\s*(?:is|:)?\s*(\d+)\b", t)
    if m3:
        return _to_int_safe(m3.group(1))
    return None

def _extract_type(text: str) -> Optional[str]:
    t = text.lower()
    m = re.search(r"\btype\s*(?:is|:)?\s*([a-zA-Z]+)\b", t)
    if m:
        return m.group(1).strip()
    for kw in ["electric", "gas", "solar", "battery", "diesel"]:
        if re.search(rf"\b{kw}\b", t):
            return kw
    return None

def _extract_location(text: str) -> Optional[str]:
    t = text.lower()
    for loc in KNOWN_LOCATIONS:
        if re.search(rf"\b{re.escape(loc)}\b", t):
            return loc
    m = re.search(r"\b(?:in|for|at)\s+(?:the\s+)?([a-z][a-z ]{1,30})\b", t)
    if m:
        candidate = m.group(1).strip()
        candidate = re.sub(r"\b(device|appliance|room|area)\b.*$", "", candidate).strip()
        words = candidate.split()
        candidate = " ".join(words[:3])
        return candidate if candidate else None
    return None

def _extract_device_name(text: str) -> Optional[str]:
    """
    Extract device name by detecting brand + appliance keyword pairs.
    """
    t = text.lower()
    words = re.findall(r"[a-z0-9]+", t)
    
    # Look for brand+device pattern
    for i in range(len(words)-1):
        two_words = f"{words[i]} {words[i+1]}"
        three_words = " ".join(words[i:i+3])
        for keyword in DEVICE_KEYWORDS:
            if keyword in two_words or keyword in three_words:
                # Include brand before the keyword if available
                start = max(0, i-1)
                candidate = " ".join(words[start:i+3])
                return candidate.strip()
    return None


def parse_details(text: str) -> DeviceDetails:
    return DeviceDetails(
        name=_extract_device_name(text),
        type=_extract_type(text),
        powerwatts=_extract_power_watts(text),
        rating=_extract_rating(text),
        location=_extract_location(text)
    )

def speech_to_text(audio_path: str) -> str:
    try:
        result = whisper_model.transcribe(audio_path)
        return result.get("text", "").strip()
    except Exception as e:
        print(f"[ERROR] Whisper transcription failed: {e}")
        return ""

app = FastAPI(title="GreenMeter Voice API")


@app.post("/parse", response_model=DeviceDetails)
def parse_text(req: ParseRequest):
    return parse_details(req.text)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, file.filename)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    try:
        text = speech_to_text(tmp_path)
        if not text:
            return {"error": "Transcription failed or empty audio"}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return {"text": text}

@app.post("/add-device-voice", response_model=DeviceDetails)
async def add_device_voice(file: UploadFile = File(...)):
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, file.filename)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    try:
        text = speech_to_text(tmp_path)
        details = parse_details(text) if text else DeviceDetails()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return details
