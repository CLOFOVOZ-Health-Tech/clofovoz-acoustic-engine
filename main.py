import os
import uuid
import hmac
import hashlib
import logging
import math
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Request
from pydantic import BaseModel
from supabase import create_client, Client
import numpy as np
import librosa
import requests

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(title="CLOFOVOZ Acoustic AI Engine")

# Endpoint de Health Check
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "engine": "CLOFOVOZ Acoustic AI Engine",
        "version": "1.0-mvp"
    }

# Inicialización de variables de entorno
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
POSTHOG_API_KEY = os.getenv("POSTHOG_API_KEY")

# Inicializar cliente de Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Modelo para el payload
class ProcessRequest(BaseModel):
    session_id: str
    user_id: str
    r2_file_key: str

# Función para descargar audio de R2
def download_audio_from_r2(file_key: str, local_path: str):
    """Descarga archivo desde Cloudflare R2"""
    try:
        # Usar la URL pública de R2
        url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com/{R2_BUCKET_NAME}/{file_key}"
        
        # Para R2 necesitamos autenticación
        import boto3
        s3_client = boto3.client(
            's3',
            endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name="auto"
        )
        s3_client.download_file(R2_BUCKET_NAME, file_key, local_path)
        logger.info(f"✅ Audio descargado: {local_path}")
    except Exception as e:
        logger.error(f"❌ Error descargando de R2: {str(e)}")
        raise

# Función para procesar audio con FFmpeg
def process_audio_to_standard(input_path: str, output_path: str):
    """Convierte audio a WAV 16kHz Mono usando FFmpeg"""
    import subprocess
    
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"✅ Audio procesado: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error en FFmpeg: {e.stderr}")
        raise RuntimeError(f"Fallo al procesar audio: {e.stderr}")

# Función para extraer características acústicas
def extract_acoustic_features(audio_path: str) -> dict:
    """Extrae F0, Jitter, Shimmer del audio"""
    try:
        # Cargar audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Calcular F0 (frecuencia fundamental)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        f0_clean = f0[~np.isnan(f0)]
        f0_mean = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
        
        # Métricas básicas (Jitter/Shimmer requieren parselmouth, simplificado aquí)
        return {
            "f0_mean_hz": round(f0_mean, 2),
            "duration_seconds": round(len(y) / sr, 2),
            "is_voiced": len(f0_clean) > (len(y) / sr) * 0.3
        }
    except Exception as e:
        logger.error(f"❌ Error extrayendo features: {str(e)}")
        return {"f0_mean_hz": 0.0, "duration_seconds": 0.0, "is_voiced": False}

# Función para análisis de IA con Hugging Face
def get_ai_vocal_insight(audio_path: str, features: dict) -> dict:
    """Envía audio a Hugging Face para análisis"""
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        # Usar modelo de clasificación de audio
        response = requests.post(
            "https://api-inference.huggingface.co/models/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            headers=headers,
            data=audio_bytes
        )
        
        if response.status_code == 200:
            result = response.json()
            top_prediction = result[0]
            return {
                "ai_classification": top_prediction['label'].lower(),
                "confidence_score": round(top_prediction['score'], 2),
                "recommendation": "Análisis completado"
            }
        else:
            logger.warning(f"HF API returned {response.status_code}")
            return {
                "ai_classification": "unknown",
                "confidence_score": 0.0,
                "recommendation": "No se pudo completar el análisis de IA"
            }
    except Exception as e:
        logger.error(f"❌ Error en Hugging Face: {str(e)}")
        return {
            "ai_classification": "unknown",
            "confidence_score": 0.0,
            "recommendation": "Error en análisis de IA"
        }

# Pipeline principal
async def run_acoustic_pipeline(session_id: str, user_id: str, r2_file_key: str):
    raw_webm_path = f"/tmp/{uuid.uuid4()}_raw.webm"
    processed_wav_path = f"/tmp/{uuid.uuid4()}_processed.wav"
    
    try:
        logger.info(f"🚀 Iniciando pipeline para sesión {session_id}")
        
        # A. Descargar de R2
        logger.info(f"📥 Descargando audio de R2: {r2_file_key}")
        download_audio_from_r2(r2_file_key, raw_webm_path)
        
        # B. Procesar audio
        logger.info("🔧 Procesando audio con FFmpeg...")
        process_audio_to_standard(raw_webm_path, processed_wav_path)
        
        # C. Extraer métricas
        logger.info("📊 Extrayendo características acústicas...")
        acoustic_features = extract_acoustic_features(processed_wav_path)
        logger.info(f"✅ Métricas: F0={acoustic_features['f0_mean_hz']}Hz")
        
        # D. Análisis IA
        logger.info("🤖 Ejecutando análisis de IA...")
        ai_insight = get_ai_vocal_insight(processed_wav_path, acoustic_features)
        
        # E. Actualizar Supabase
        logger.info("💾 Guardando en Supabase...")
        supabase.table("vocal_sessions").update({
            "status": "completed",
            "metrics": acoustic_features,
            "ai_insight": ai_insight,
            "completed_at": "now()"
        }).eq("id", session_id).execute()
        
        logger.info(f"✅ Pipeline completado: {session_id}")

    except Exception as e:
        logger.error(f"❌ Pipeline falló: {str(e)}")
        supabase.table("vocal_sessions").update({
            "status": "failed",
            "error_message": str(e)[:500]
        }).eq("id", session_id).execute()
        
    finally:
        # Limpieza
        for path in [raw_webm_path, processed_wav_path]:
            if os.path.exists(path):
                os.remove(path)

# Endpoint principal del webhook
@app.post("/api/v1/process-session")
async def process_vocal_session(
    request: Request,
    background_tasks: BackgroundTasks,
    webhook_secret: str | None = Header(None, alias="Webhook-Secret")
):
    # Validar secreto
    if not webhook_secret or not hmac.compare_digest(webhook_secret, WEBHOOK_SECRET or ""):
        logger.warning("❌ Webhook secret mismatch")
        raise HTTPException(status_code=401, detail="Webhook secret mismatch")

    # Parsear payload
    payload = await request.json()
    logger.info(f"📩 Payload recibido: {payload}")
    
    if payload.get("type") != "INSERT" or payload.get("table") != "vocal_sessions":
        raise HTTPException(status_code=400, detail="Invalid event type or table")

    record = payload.get("record", {})
    session_id = record.get("id")
    user_id = record.get("user_id")
    r2_file_key = record.get("r2_file_key")

    if not all([session_id, user_id, r2_file_key]):
        raise HTTPException(status_code=400, detail="Missing required fields")

    # Disparar en background
    logger.info(f"🚀 Webhook recibido para sesión {session_id}")
    background_tasks.add_task(run_acoustic_pipeline, session_id, user_id, r2_file_key)
    
    return {"status": "accepted", "session_id": session_id}

# Para desarrollo local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)