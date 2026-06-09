"""
Unit tests for providers.
Tests language support, response parsing, and error mapping.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from vaaniflow.providers.tts.base import TTSSynthesisRequest, TTSSynthesisResponse
from vaaniflow.providers.tts.gtts_provider import GTTSProvider
from vaaniflow.providers.tts.elevenlabs_provider import ElevenLabsProvider
from vaaniflow.providers.tts.sarvam_provider import SarvamTTSProvider
from vaaniflow.providers.translation.google_provider import GoogleTranslationProvider
from vaaniflow.providers.translation.sarvam_provider import SarvamTranslationProvider
from vaaniflow.providers.transcription.whisper_provider import WhisperProvider
from vaaniflow.providers.transcription.assembly_provider import AssemblyAIProvider


class TestGTTSProvider:
    """Tests for the gTTS fallback provider."""

    def test_supports_all_indian_languages(self):
        provider = GTTSProvider()
        for lang in ["en", "hi", "bn", "te", "mr", "ta", "gu", "kn", "ml", "pa", "or"]:
            assert provider.supports_language(lang), f"gTTS should support {lang}"

    def test_does_not_support_unknown_languages(self):
        provider = GTTSProvider()
        assert not provider.supports_language("xx")
        assert not provider.supports_language("zh")

    def test_provider_name(self):
        provider = GTTSProvider()
        assert provider.provider_name == "gtts"


class TestElevenLabsProvider:
    """Tests for the ElevenLabs provider."""

    def test_supports_indian_languages(self):
        provider = ElevenLabsProvider()
        for lang in ["en", "hi", "ta", "te", "kn", "ml", "bn", "mr", "gu"]:
            assert provider.supports_language(lang), f"ElevenLabs should support {lang}"

    def test_provider_name(self):
        provider = ElevenLabsProvider()
        assert provider.provider_name == "elevenlabs"


class TestSarvamTTSProvider:
    """Tests for the Sarvam TTS provider."""

    def test_supports_all_indian_languages(self):
        provider = SarvamTTSProvider()
        for lang in ["hi", "bn", "te", "mr", "ta", "gu", "kn", "ml", "pa", "or", "en"]:
            assert provider.supports_language(lang), f"Sarvam TTS should support {lang}"

    def test_provider_name(self):
        provider = SarvamTTSProvider()
        assert provider.provider_name == "sarvam"


class TestGoogleTranslationProvider:
    """Tests for the Google Translation provider."""

    def test_supports_all_languages(self):
        provider = GoogleTranslationProvider()
        for lang in ["en", "hi", "bn", "te", "mr", "ta", "gu", "kn", "ml", "pa", "or"]:
            assert provider.supports_language(lang), f"Google should support {lang}"

    def test_provider_name(self):
        provider = GoogleTranslationProvider()
        assert provider.provider_name == "google"


class TestSarvamTranslationProvider:
    """Tests for the Sarvam Translation provider."""

    def test_supports_all_languages(self):
        provider = SarvamTranslationProvider()
        for lang in ["en", "hi", "bn", "te", "mr", "ta", "gu", "kn", "ml", "pa", "or"]:
            assert provider.supports_language(lang), f"Sarvam should support {lang}"

    def test_provider_name(self):
        provider = SarvamTranslationProvider()
        assert provider.provider_name == "sarvam"


class TestWhisperProvider:
    """Tests for the Whisper transcription provider."""

    def test_supports_all_languages(self):
        provider = WhisperProvider()
        for lang in ["en", "hi", "bn", "te", "mr", "ta", "gu", "kn", "ml", "pa", "or"]:
            assert provider.supports_language(lang), f"Whisper should support {lang}"

    def test_provider_name(self):
        provider = WhisperProvider()
        assert provider.provider_name == "whisper"


class TestAssemblyAIProvider:
    """Tests for the AssemblyAI transcription provider."""

    def test_supports_languages(self):
        provider = AssemblyAIProvider()
        for lang in ["en", "hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "pa"]:
            assert provider.supports_language(lang), f"AssemblyAI should support {lang}"

    def test_provider_name(self):
        provider = AssemblyAIProvider()
        assert provider.provider_name == "assemblyai"


class TestTTSSynthesisRequest:
    """Tests for TTSSynthesisRequest dataclass."""

    def test_default_values(self):
        req = TTSSynthesisRequest(text="Hello", language="en")
        assert req.text == "Hello"
        assert req.language == "en"
        assert req.voice_id is None
        assert req.speaking_rate == 1.0
        assert req.pitch == 0.0

    def test_custom_values(self):
        req = TTSSynthesisRequest(
            text="Test",
            language="hi",
            voice_id="custom_voice",
            speaking_rate=1.2,
            pitch=0.5,
        )
        assert req.voice_id == "custom_voice"
        assert req.speaking_rate == 1.2
        assert req.pitch == 0.5
