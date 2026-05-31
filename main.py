import os
import math
import numpy as np
import librosa
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from supabase import create_client, Client

app = FastAPI(title="CLOFOVOZ Acoustic AI Engine")

# Endpoint de Salud obligatorio para el Health Check de Railway
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

# Inicialización clásica y segura exigida por la versión 2.11.0
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

class AnalysisRequest(BaseModel):
    audio_id: str
    r2_key: str
    user_id: str

def analyze_vocal_acoustics(audio_id: str, r2_key: str, user_id: str):
    temp_filename = f"temp_{audio_id}.wav"
    try:
        audio_url = f"{R2_PUBLIC_URL}/{r2_key}"
        response = requests.get(audio_url, stream=True)
        if response.status_code != 200:
            print(f"Error descargas R2. Status: {response.status_code}")
            return

        with open(temp_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        y, sr = librosa.load(temp_filename, sr=None)

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr
        )
        f0_clean = f0[~np.isnan(f0)]

        if len(f0_clean) == 0:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return

        pitch_mean = float(np.mean(f0_clean))
        pitch_min = float(np.min(f0_clean))
        pitch_max = float(np.max(f0_clean))

        differences_f0 = np.abs(np.diff(f0_clean))
        jitter = float(np.mean(differences_f0) / pitch_mean) if pitch_mean > 0 else 0

        rms = librosa.feature.rms(y=y)[0]
        rms_clean = rms[rms > np.max(rms) * 0.01]
        if len(rms_clean) > 1:
            shimmer = float(np.mean(np.abs(np.diff(rms_clean))) / np.mean(rms_clean))
        else:
            shimmer = 0

        autocorr = librosa.autocorrelate(y)
        r_0 = autocorr[0]
        r_max = np.max(autocorr[1:]) if len(autocorr) > 1 else 1
        hnr = float(10 * math.log10(r_max / (r_0 - r_max))) if (r_0 - r_max) > 0 else 0

        score_base = 10.0
        penalizacion_jitter = (jitter * 100) * 2.5
        penalizacion_shimmer = (shimmer * 100) * 1.5
        score_coherencia = max(0.0, min(10.0, score_base - penalizacion_jitter - penalizacion_shimmer + (hnr / 10)))

        supabase.table("vocal_biomarkers").insert({
            "audio_id": audio_id,
            "user_id": user_id,
            "jitter": round(jitter, 4),
            "shimmer": round(shimmer, 4),
            "hnr": round(hnr, 3),
            "pitch_mean": round(pitch_mean, 2),
            "pitch_min": round(pitch_min, 2),
            "pitch_max": round(pitch_max, 2),
            "score_coherencia": round(score_coherencia, 2),
            "raw_analysis_json": {"engine_version": "v1.0-mvp", "samples_processed": len(f0)}
        }).execute()

    except Exception as e:
        print(f"Error crítico: {str(e)}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/api/v1/analyze")
async def trigger_analysis(payload: AnalysisRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        analyze_vocal_acoustics, 
        payload.audio_id, 
        payload.r2_key, 
        payload.user_id
    )
    return {"status": "processing", "message": "Análisis iniciado."}

if __name__ == "__main__":
    import uvicorn
    # Leer puerto dinámico de la nube para el balanceador de Railway
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)