"""
Voices API endpoint — list available voices for TTS providers.
"""
from fastapi import APIRouter, Query
from typing import Optional
from vaaniflow.models import TTSProvider

router = APIRouter()

VOICES_CATALOG = {
    TTSProvider.SARVAM: [
        {"id": "arvind", "name": "Arvind", "gender": "Male", "description": "Male - Clear & Authoritative (Hindi, Bengali, Marathi, etc.)"},
        {"id": "amartya", "name": "Amartya", "gender": "Male", "description": "Male - Warm & Conversational"},
        {"id": "ratan", "name": "Ratan", "gender": "Male", "description": "Male - Deep & Formal"},
        {"id": "meera", "name": "Meera", "gender": "Female", "description": "Female - Expressive & Natural (Telugu, Tamil, Kannada, etc.)"},
        {"id": "kavya", "name": "Kavya", "gender": "Female", "description": "Female - Soft & Professional"},
        {"id": "anvita", "name": "Anvita", "gender": "Female", "description": "Female - Bright & Friendly"},
    ],
    TTSProvider.ELEVENLABS: [
        {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam", "gender": "Male", "description": "Male - Deep & Versatile"},
        {"id": "ErXwobaYiN019PkySvjV", "name": "Antoni", "gender": "Male", "description": "Male - Well-rounded"},
        {"id": "VR6AewLTigWG4xTvoXx4", "name": "Arnold", "gender": "Male", "description": "Male - Crisp & Resonant"},
        {"id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "gender": "Female", "description": "Female - Calm & Professional"},
        {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella", "gender": "Female", "description": "Female - Expressive"},
        {"id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi", "gender": "Female", "description": "Female - Strong & Clear"},
    ],
    TTSProvider.GTTS: [
        {"id": "default", "name": "Google Default (Female)", "gender": "Female", "description": "Google Standard Web Voice"},
    ],
}


@router.get("/")
async def list_voices(
    provider: Optional[TTSProvider] = Query(default=None),
    gender: Optional[str] = Query(default=None),
):
    """
    List available voices per provider.
    Supports filtering by provider and speaker gender.
    """
    if provider:
        voices = VOICES_CATALOG.get(provider, [])
    else:
        # Return dict grouped by provider
        result = {}
        for prov, prov_voices in VOICES_CATALOG.items():
            if gender:
                result[prov.value] = [v for v in prov_voices if v["gender"].lower() == gender.lower()]
            else:
                result[prov.value] = prov_voices
        return {"voices": result}

    if gender:
        voices = [v for v in voices if v["gender"].lower() == gender.lower()]

    return {
        "provider": provider.value,
        "count": len(voices),
        "voices": voices,
    }
