import os
import shutil
from pathlib import Path
from typing import Optional, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from inspector import inspect_image


class InspectionResponse(BaseModel):
    status: str
    result: Optional[str]
    message: str
    output_image: Optional[str] = None
    report: Optional[Any] = None


app = FastAPI(
    title="AOI Inspection API",
    version="1.0.0",
    description="Empfängt ein Bild und liefert IO/NIO zurück."
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {"message": "AOI API läuft"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inspect", response_model=InspectionResponse)
def inspect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Nur Bilddateien sind erlaubt")

    original_name = file.filename or "upload.png"
    safe_name = Path(original_name).name
    save_path = UPLOAD_DIR / safe_name

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = inspect_image(str(save_path))

    if result["status"] == "ERROR":
        raise HTTPException(status_code=500, detail=result["message"])

    return result