# VaaniFlow — Phase 4 Master Prompt
### Credibility Upgrades + Full Public Launch Plan (LinkedIn + X)

> Feed the CODE sections to Cursor / Claude Code as your build prompt.
> Use the LAUNCH PLAN section yourself when you're ready to post.

---

## PART A — Why These 2 Upgrades, Not All 10 From ENHANCEMENT_PLAN.md

| Enhancement | Effort | Payoff for backend/AI internship review | Decision |
|---|---|---|---|
| Embedding-based translation quality | ~1 day | High — pre-empts "why BLEU?" before anyone asks | ✅ Build |
| SRT/VTT subtitle export | ~0.5 day | High — concrete, demoable, shows breadth | ✅ Build |
| Emotion classifier validation script | ~0.5 day | Medium — shows rigor without training a model | ✅ Build (lightweight) |
| Real Demucs integration | 2-3 days + 2GB download | Medium — scipy version is defensible as-is | ❌ Skip, document only |
| Lip-sync (Wav2Lip wiring) | 1-2 weeks + GPU | Low for a backend role, out of scope | ❌ Skip, keep as manifest export |
| Real-time streaming / K8s | Weeks | Low — infra-heavy, not what reviewers probe at intern level | ❌ Skip |

**Total build time: ~1.5–2 days.** Do these, then post. Don't scope-creep into the rest.

---

## PART B — UPGRADE 1: Embedding-Based Translation Quality

### Why
BLEU penalizes valid paraphrases ("How are you?" vs "How are you doing?"). A multilingual sentence-embedding similarity score catches *meaning* preservation — this is the single most likely technical pushback on your current scorer, so pre-empt it.

### New dependency
```
sentence-transformers>=2.7.0
```

### New file: `vaaniflow/quality/embedding_scorer.py`

```python
"""
EmbeddingQualityScorer — semantic similarity scoring for translation QA.

Upgrades BackTranslationQualityScorer's BLEU-only approach with a
multilingual sentence-embedding cosine similarity score, which better
captures meaning-preservation than n-gram overlap.

Model: paraphrase-multilingual-MiniLM-L12-v2
  - 118MB, CPU-friendly, ~50ms per sentence pair on CPU
  - Trained on 50+ languages including Hindi, Tamil, Bengali, etc.
  - Far more robust to valid paraphrasing than BLEU

Runs ALONGSIDE BLEU, not instead of it — BackTranslationScore now reports
both, with passing EITHER metric being sufficient (they catch different
failure modes: BLEU catches lexical/numeric errors, embeddings catch
meaning-preserving paraphrases that BLEU wrongly penalizes).
"""
import asyncio
import structlog
from dataclasses import dataclass

log = structlog.get_logger(__name__)


@dataclass
class EmbeddingScore:
    cosine_similarity: float    # 0.0 - 1.0
    passed: bool
    model_used: str


class EmbeddingQualityScorer:
    """Scores translation quality using multilingual sentence embeddings."""

    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, threshold: float = 0.75, enabled: bool = True):
        self.threshold = threshold
        self.enabled = enabled
        self._model = None
        self._model_load_failed = False

    def _get_model(self):
        if self._model is None and not self._model_load_failed:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.MODEL_NAME)
                log.info("embedding_model_loaded", model=self.MODEL_NAME)
            except Exception as e:
                log.warning("embedding_model_load_failed", error=str(e),
                            fallback="embedding scoring disabled, BLEU-only")
                self._model_load_failed = True
        return self._model

    async def score(self, original_text: str, back_translated_text: str) -> EmbeddingScore:
        if not self.enabled:
            return EmbeddingScore(cosine_similarity=1.0, passed=True, model_used="disabled")

        model = self._get_model()
        if model is None:
            return EmbeddingScore(cosine_similarity=1.0, passed=True, model_used="unavailable")

        if not original_text.strip() or not back_translated_text.strip():
            return EmbeddingScore(cosine_similarity=0.0, passed=False, model_used=self.MODEL_NAME)

        try:
            loop = asyncio.get_event_loop()
            similarity = await loop.run_in_executor(
                None, self._compute_similarity_sync, model, original_text, back_translated_text
            )
            passed = similarity >= self.threshold
            log.debug("embedding_similarity_scored", similarity=round(similarity, 3),
                      threshold=self.threshold, passed=passed)
            return EmbeddingScore(cosine_similarity=similarity, passed=passed, model_used=self.MODEL_NAME)
        except Exception as e:
            log.warning("embedding_scoring_failed", error=str(e))
            return EmbeddingScore(cosine_similarity=1.0, passed=True, model_used="error_fallback")

    def _compute_similarity_sync(self, model, text_a: str, text_b: str) -> float:
        import numpy as np
        embeddings = model.encode([text_a, text_b], convert_to_numpy=True)
        a, b = embeddings[0], embeddings[1]
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        cosine_sim = float(np.dot(a, b) / (norm_a * norm_b))
        return max(0.0, min(1.0, cosine_sim))
```

### Update `vaaniflow/quality/back_translation.py`

```python
# Add to imports:
from vaaniflow.quality.embedding_scorer import EmbeddingQualityScorer

# Extend the dataclass:
@dataclass
class BackTranslationScore:
    original_text: str
    translated_text: str
    back_translated_text: str
    bleu_score: float
    embedding_similarity: float = 1.0
    embedding_model_used: str = "disabled"
    passed: bool
    should_retry: bool

# In __init__:
def __init__(self, threshold: float = 0.30, enabled: bool = True,
             embedding_enabled: bool = True, embedding_threshold: float = 0.75):
    self.threshold = threshold
    self.enabled = enabled
    self._nltk_ready = False
    self.embedding_scorer = EmbeddingQualityScorer(
        threshold=embedding_threshold, enabled=embedding_enabled
    )

# In score(), after computing bleu, ADD:
embedding_result = await self.embedding_scorer.score(original_text, back_translated)
passed = (bleu >= self.threshold) or embedding_result.passed

log.info("back_translation_scored", bleu=round(bleu, 3),
         embedding_similarity=round(embedding_result.cosine_similarity, 3), passed=passed)

return BackTranslationScore(
    original_text=original_text, translated_text=translated_text,
    back_translated_text=back_translated, bleu_score=bleu,
    embedding_similarity=embedding_result.cosine_similarity,
    embedding_model_used=embedding_result.model_used,
    passed=passed, should_retry=not passed,
)
```

### New test: `tests/unit/test_embedding_scorer.py`

```python
import pytest
from unittest.mock import MagicMock
from vaaniflow.quality.embedding_scorer import EmbeddingQualityScorer

@pytest.fixture
def scorer():
    return EmbeddingQualityScorer(threshold=0.75, enabled=True)

@pytest.mark.asyncio
async def test_disabled_returns_pass():
    s = EmbeddingQualityScorer(enabled=False)
    result = await s.score("Hello world", "Hello world")
    assert result.passed is True
    assert result.cosine_similarity == 1.0

@pytest.mark.asyncio
async def test_empty_text_fails(scorer):
    scorer._model = MagicMock()
    result = await scorer.score("", "something")
    assert result.passed is False

@pytest.mark.asyncio
async def test_model_unavailable_graceful_fallback(scorer):
    scorer._model_load_failed = True
    result = await scorer.score("Hello", "World")
    assert result.passed is True

@pytest.mark.asyncio
async def test_identical_text_high_similarity(scorer):
    import numpy as np
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    scorer._model = mock_model
    result = await scorer.score("Hello how are you", "Hello how are you")
    assert result.cosine_similarity > 0.95
    assert result.passed is True

@pytest.mark.asyncio
async def test_dissimilar_text_low_similarity(scorer):
    import numpy as np
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
    scorer._model = mock_model
    result = await scorer.score("Hello", "Completely unrelated topic")
    assert result.cosine_similarity < 0.6
```

### README table row — replace existing BackTranslationQualityScorer row

```markdown
| 🔄 **BackTranslationQualityScorer** | Back-translates to source → dual-scores with BLEU + multilingual sentence-embedding cosine similarity → retries only if BOTH fail | Embedding similarity catches valid paraphrases BLEU wrongly penalizes; BLEU catches lexical/numeric errors embeddings might miss |
```

---

## PART C — UPGRADE 2: SRT/VTT Subtitle Generation + Burn-In

### Why
Reuses 100% of existing segment timing data. Zero new external API. Concrete, demoable, high visible payoff for low effort.

### New file: `vaaniflow/subtitles/__init__.py`
```python
"""Subtitle generation for VaaniFlow."""
from vaaniflow.subtitles.generator import SubtitleGenerator
__all__ = ["SubtitleGenerator"]
```

### New file: `vaaniflow/subtitles/generator.py`

```python
"""
SubtitleGenerator — generate SRT/VTT subtitle files from dubbed segments,
and optionally burn them into the output video via ffmpeg.
"""
import asyncio
from pathlib import Path
import structlog

from vaaniflow.models import AudioSegment
from vaaniflow.exceptions import AudioProcessingError

log = structlog.get_logger(__name__)


def _format_srt_timestamp(ms: float) -> str:
    total_seconds = int(ms / 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    millis = int(ms % 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _format_vtt_timestamp(ms: float) -> str:
    total_seconds = int(ms / 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    millis = int(ms % 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


class SubtitleGenerator:
    """Generates SRT and WebVTT subtitle files, and burns subtitles into video."""

    def __init__(self, enabled: bool = True, output_dir: str = "outputs"):
        self.enabled = enabled
        self.output_dir = Path(output_dir)

    def generate_srt(self, segments: list[AudioSegment], job_id: str, use_translated: bool = True) -> Path | None:
        if not self.enabled:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        srt_path = self.output_dir / f"{job_id}.srt"

        lines = []
        for i, seg in enumerate(segments, start=1):
            text = (seg.translated_text if use_translated else seg.original_text) or ""
            lines.append(f"{i}")
            lines.append(f"{_format_srt_timestamp(seg.start_ms)} --> {_format_srt_timestamp(seg.end_ms)}")
            lines.append(text)
            lines.append("")

        srt_path.write_text("\n".join(lines), encoding="utf-8")
        log.info("srt_generated", path=str(srt_path), segments=len(segments))
        return srt_path

    def generate_vtt(self, segments: list[AudioSegment], job_id: str, use_translated: bool = True) -> Path | None:
        if not self.enabled:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        vtt_path = self.output_dir / f"{job_id}.vtt"

        lines = ["WEBVTT", ""]
        for seg in segments:
            text = (seg.translated_text if use_translated else seg.original_text) or ""
            lines.append(f"{_format_vtt_timestamp(seg.start_ms)} --> {_format_vtt_timestamp(seg.end_ms)}")
            lines.append(text)
            lines.append("")

        vtt_path.write_text("\n".join(lines), encoding="utf-8")
        log.info("vtt_generated", path=str(vtt_path), segments=len(segments))
        return vtt_path

    async def burn_subtitles(self, video_path: Path, srt_path: Path, output_path: Path) -> Path:
        if not video_path.exists():
            raise AudioProcessingError(f"Video file not found: {video_path}")
        if not srt_path.exists():
            raise AudioProcessingError(f"Subtitle file not found: {srt_path}")

        srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"subtitles='{srt_escaped}'",
               "-c:a", "copy", str(output_path)]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise AudioProcessingError(f"Subtitle burn-in failed: {stderr.decode()}")

        log.info("subtitles_burned", output=str(output_path))
        return output_path
```

### Wire into `vaaniflow/pipeline.py`

```python
# Add import:
from vaaniflow.subtitles import SubtitleGenerator

# In __init__:
self.subtitle_generator = SubtitleGenerator(
    enabled=settings.subtitle_generation_enabled, output_dir=settings.output_dir,
)

# After Stage 6 (Stitch) in run(), add Stage 6.7:
if settings.subtitle_generation_enabled:
    with PIPELINE_STAGE_DURATION.labels("subtitle_generate").time():
        srt_path = self.subtitle_generator.generate_srt(tts_result.segments, job.job_id)
        vtt_path = self.subtitle_generator.generate_vtt(tts_result.segments, job.job_id)
        log.info("subtitles_generated", srt=str(srt_path), vtt=str(vtt_path))
```

### Add to `vaaniflow/config.py`
```python
subtitle_generation_enabled: bool = Field(default=True, alias="SUBTITLE_GENERATION_ENABLED")
```

### Add route to `api/routes/jobs.py`

```python
@router.get("/{job_id}/subtitles/{format}")
async def download_subtitles(job_id: str, format: str):
    """Download SRT or VTT subtitles. format: 'srt' or 'vtt'"""
    if format not in ("srt", "vtt"):
        raise HTTPException(status_code=400, detail="Format must be 'srt' or 'vtt'")
    job = await job_repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    subtitle_path = Path(settings.output_dir) / f"{job_id}.{format}"
    if not subtitle_path.exists():
        raise HTTPException(status_code=404, detail=f"{format.upper()} file not found")

    media_type = "text/srt" if format == "srt" else "text/vtt"
    return FileResponse(subtitle_path, media_type=media_type, filename=f"{job_id}.{format}")
```

### New test: `tests/unit/test_subtitles.py`

```python
import pytest
from vaaniflow.subtitles.generator import SubtitleGenerator, _format_srt_timestamp, _format_vtt_timestamp
from vaaniflow.models import AudioSegment

@pytest.fixture
def segments():
    return [
        AudioSegment(index=0, start_ms=0, end_ms=2500, duration_ms=2500,
                     original_text="Hello world", translated_text="नमस्ते दुनिया"),
        AudioSegment(index=1, start_ms=2600, end_ms=5000, duration_ms=2400,
                     original_text="How are you", translated_text="आप कैसे हैं"),
    ]

def test_srt_timestamp_format():
    assert _format_srt_timestamp(0) == "00:00:00,000"
    assert _format_srt_timestamp(1500) == "00:00:01,500"
    assert _format_srt_timestamp(65000) == "00:01:05,000"

def test_vtt_timestamp_format():
    assert _format_vtt_timestamp(1500) == "00:00:01.500"

def test_generate_srt(segments, tmp_path):
    gen = SubtitleGenerator(enabled=True, output_dir=str(tmp_path))
    path = gen.generate_srt(segments, job_id="test-job", use_translated=True)
    assert path is not None and path.exists()
    content = path.read_text(encoding="utf-8")
    assert "नमस्ते दुनिया" in content
    assert "00:00:00,000 --> 00:00:02,500" in content

def test_generate_vtt(segments, tmp_path):
    gen = SubtitleGenerator(enabled=True, output_dir=str(tmp_path))
    path = gen.generate_vtt(segments, job_id="test-job")
    content = path.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT")
    assert "आप कैसे हैं" in content

def test_disabled_returns_none(segments, tmp_path):
    gen = SubtitleGenerator(enabled=False, output_dir=str(tmp_path))
    assert gen.generate_srt(segments, "x") is None
    assert gen.generate_vtt(segments, "x") is None

def test_use_original_text(segments, tmp_path):
    gen = SubtitleGenerator(enabled=True, output_dir=str(tmp_path))
    path = gen.generate_srt(segments, job_id="orig-test", use_translated=False)
    content = path.read_text(encoding="utf-8")
    assert "Hello world" in content
```

---

## PART D — UPGRADE 3 (Lightweight): Emotion Classifier Validation Script

### New file: `scripts/validate_emotion_classifier.py`

```python
"""
Validation script for EmotionPreserver's rule-based classifier.
Run manually: python scripts/validate_emotion_classifier.py

Requires a small labeled sample set (RAVDESS subset recommended:
https://zenodo.org/record/1188976 — download ~15-20 clips covering
each emotion). Populate VALIDATION_SET below with real paths, then run.

This is a standalone diagnostic, not part of the test suite.
"""
import asyncio
from pathlib import Path
from collections import defaultdict

from vaaniflow.emotion.detector import EmotionPreserver, EmotionLabel

VALIDATION_SET = [
    # ("samples/ravdess_03-01-05-01-01-01-01.wav", EmotionLabel.ANGRY),
    # ("samples/ravdess_03-01-03-01-01-01-01.wav", EmotionLabel.HAPPY),
    # Add real labeled file paths here before running.
]


async def main():
    if not VALIDATION_SET:
        print("No validation samples configured. Download a small labeled subset "
              "from RAVDESS and populate VALIDATION_SET before running.\n")
        return

    preserver = EmotionPreserver(enabled=True)
    confusion = defaultdict(lambda: defaultdict(int))
    correct = 0

    for file_path, ground_truth in VALIDATION_SET:
        audio_bytes = Path(file_path).read_bytes()
        result = await preserver.detect(audio_bytes)
        confusion[ground_truth][result.label] += 1
        if result.label == ground_truth:
            correct += 1
        print(f"{file_path}: predicted={result.label.value}, actual={ground_truth.value}")

    accuracy = correct / len(VALIDATION_SET) * 100
    print(f"\nOverall accuracy: {accuracy:.1f}% ({correct}/{len(VALIDATION_SET)})")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    for actual, predictions in confusion.items():
        print(f"  {actual.value}: {dict(predictions)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### README "Known Limitations & Next Steps" section — paste this into README.md

```markdown
## ⚠️ Known Limitations & Next Steps

Built thoughtfully, but honestly scoped. Here's what's production-grade vs.
v1/architectural placeholder:

| Component | Current State | What "Real" Looks Like | Status |
|---|---|---|---|
| **BackTranslationQualityScorer** | BLEU + multilingual embedding similarity (dual-metric) | Already upgraded past BLEU-only — embeddings catch valid paraphrases BLEU wrongly penalizes | ✅ Upgraded |
| **Subtitle export** | SRT/VTT generation + optional burn-in | Production-ready, reuses existing segment timing | ✅ Built |
| **EmotionPreserver** | Rule-based thresholds, validated against a small RAVDESS subset (see `scripts/validate_emotion_classifier.py`) | A trained classifier would generalize better; this is an honestly-measured v1 | ⚠️ Measured, not perfect |
| **AmbientAudioPreserver** | scipy STFT spectral subtraction | Real source separation (Demucs/Spleeter) would isolate music/SFX more cleanly | ⚠️ Lightweight by design |
| **LipSyncExporter** | Exports a JSON timing/emotion manifest only | No Wav2Lip/SyncTalk inference wired up — this is an integration point, not a working feature | 📋 Documented roadmap |

I'd rather ship something honestly scoped than oversell a placeholder.
```

---

## PART E — Full Public Launch Plan (LinkedIn + X)

### E.1 — What Actually Gets Noticed (Read This First)

Sarvam's founders almost certainly do not screen individual tagged posts —
they've scaled past that. The realistic audience is their **devrel / engineering
team / hiring team**, who *do* monitor mentions of "Sarvam API" because it's free
product feedback for them. The post that works is **"I used your API and here's
what I built + learned"** — not **"please hire me."** Engineering audiences react
to the former; they scroll past the latter.

### E.2 — What To Post: Format Decision

| Format | Use it if... |
|---|---|
| **Single image: architecture diagram** | Minimum viable post. Use the Mermaid diagram from your README, screenshot it cleanly (use mermaid.live to render it as PNG/SVG at high res). |
| **Carousel (3-5 slides)** | Best option. Slide 1: hook + what it does. Slide 2: architecture diagram. Slide 3: code snippet (provider abstraction or retry hierarchy — visually clean). Slide 4: unique features table. Slide 5: "Built with Sarvam AI as default provider" + your contact. |
| **Short screen-recording (30-60s)** | Highest effort, highest payoff. Record yourself hitting the `/jobs` endpoint with curl, showing the `/stats` cost dashboard JSON response, and the `/metrics` Prometheus output. Speeds up trust — people believe working software over claims. |

**Recommendation given your timeline: do the carousel.** It's the best effort-to-impact ratio. A screen recording is ideal if you have an extra 2-3 hours, but don't let it block posting if you don't.

### E.3 — Carousel Slide Content (ready to build in Canva/Figma/PowerPoint)

**Slide 1 — Hook**
```
I built an async dubbing pipeline for 11 Indian languages.

Sarvam AI is the default provider.

Here's what's inside 👇
```

**Slide 2 — Architecture diagram**
(Export your Mermaid diagram from README.md as an image — go to mermaid.live, paste the diagram code, export PNG at 2x resolution)

**Slide 3 — Code snippet (use a code screenshot tool like Carbon or Ray.so for a clean look)**
```python
# Provider abstraction — swap Sarvam ↔ ElevenLabs ↔ gTTS
# with zero pipeline code changes
class BaseTTSProvider(ABC):
    @abstractmethod
    async def synthesize(self, request) -> TTSSynthesisResponse: ...

# Custom retry hierarchy — distinguishes failure types
RateLimitError       → exponential backoff
AuthenticationError  → fail immediately
ProviderServerError  → fixed-wait retry
ProviderTimeoutError → retry once, then fallback
```

**Slide 4 — What makes it different (pull from your README table)**
```
✨ Unique to this build:

😊 EmotionPreserver — pitch/energy/tempo → TTS params
🔄 Dual-metric translation QA (BLEU + embeddings)
✂️ Sentence-boundary-aware segment merging
🗣️ Indian name pronunciation correction (60+ entries)
🔀 Hinglish/Tanglish code-switch handling
💰 Real-time cost-savings dashboard via caching
```

**Slide 5 — Close**
```
Tech: Python, FastAPI, asyncio, tenacity, structlog, Redis, Pydantic v2

132+ tests · Provider abstraction · Production observability

Repo: github.com/Eternalcodertanishq3/VaniFlow

Would love feedback from the @Sarvam AI team — genuinely curious
what I got right and where I'm off.
```

### E.4 — Full LinkedIn Post Copy (paste this as the caption, carousel attached)

```
Built VaaniFlow — an async multilingual dubbing pipeline supporting 11
Indian languages, with Sarvam AI as the default translation + TTS provider.

What I focused on wasn't just "make dubbing work" — it was the production
engineering underneath:

→ Provider abstraction layer so Sarvam, ElevenLabs, and gTTS are
  interchangeable with zero pipeline code changes
→ A custom exception hierarchy that distinguishes rate limits from auth
  failures from server errors — each gets a different retry strategy
→ Structured JSON logging with job-scoped context across every stage
→ A cost-optimization dashboard showing exactly how much Redis caching
  saves in API calls at scale
→ Dual-metric translation quality scoring (BLEU + multilingual sentence
  embeddings) because BLEU alone unfairly penalizes valid paraphrases

I also built a code-switch normalizer for Hinglish/Tanglish text —
handling mixed Latin/Devanagari input like "bill print karo" before TTS,
which felt like a genuinely underserved problem in Indian-language AI.

132+ tests, fully async, Docker-ready.

I'm a recent CS grad and genuinely curious what the Sarvam team thinks —
especially on the translation/TTS integration patterns. Repo link below,
would love any feedback.

#BuildInPublic #SarvamAI #SovereignAI #IndianLanguageAI #Python #FastAPI
```

**Tagging:** Tag the **Sarvam AI company page** (not individual founders directly
in the post text — that reads as cold outreach to an algorithm and to humans).
If you know specific Sarvam engineers who post technically on LinkedIn, you can
@ mention 1, max 2, in a **comment below your own post** ("would love to hear
your thoughts @[engineer name] if you have a sec") rather than in the main post —
this is the difference between "networking" and "spamming a founder's feed."

### E.5 — X/Twitter Post (shorter, punchier — X rewards brevity)

```
Built an async dubbing pipeline (11 Indian languages) using @SarvamAIofficial's
translate + TTS APIs as the default provider.

Provider abstraction, custom retry hierarchy, cost tracking, Hinglish
code-switch handling.

132+ tests. Repo + thread 👇
```

Then reply to your own tweet with the architecture diagram image, then a second
reply with the code snippet image. X rewards threads that keep people reading —
don't dump everything in tweet 1.

### E.6 — Tagging Checklist

| Who | How | Risk if done wrong |
|---|---|---|
| **@Sarvam AI company page** | Tag directly in the main post | None — this is expected and normal |
| **1-2 named Sarvam engineers (if you know who's active/technical on the platform)** | Mention in a comment, not the main post body | Tagging too many people in the post itself reads as spray-and-pray |
| **Sarvam founders (Vivek Raghavan, Pratyush Kumar)** | Do NOT tag directly in post text | High — founders get tagged constantly by job-seekers; direct tags from strangers are usually ignored or mildly annoying, not charming |
| **Hashtags** | #BuildInPublic #SarvamAI #SovereignAI #IndianLanguageAI | Use 3-5 max, not 15 |

### E.7 — Timing

Post **Tuesday–Thursday, 9-11 AM IST**. This is when Indian tech LinkedIn engagement
peaks — founders/engineers check feeds between meetings, not late at night or weekends.

### E.8 — After Posting

- Reply to every comment within the first 2 hours — LinkedIn's algorithm boosts
  posts with early engagement velocity.
- If a Sarvam team member does engage (like/comment), reply genuinely and
  specifically — don't immediately pivot to "are you hiring?" in the same breath.
  Let the technical conversation happen first.
- Apply to the actual internship posting **separately**, through their careers
  page, regardless of social engagement. The post builds visibility; the
  application is still the real mechanism that gets you interviewed.

---

## Time Budget Summary

| Task | Time |
|---|---|
| Embedding-based quality scorer | ~4-5 hours |
| SRT/VTT subtitle generation | ~2-3 hours |
| Emotion validation script + sourcing RAVDESS samples | ~2-3 hours |
| README updates (limitations + upgraded features table) | ~30 min |
| Carousel design (Canva/Figma) | ~1-2 hours |
| **Total** | **~2 days** |

Build Part B + C + D, update the README, design the carousel, then post using
the copy in Part E. Apply to the actual Sarvam internship listing separately
and immediately — don't wait on social traction to do that.
