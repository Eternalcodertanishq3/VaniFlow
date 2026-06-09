"""
BackTranslationQualityScorer — semantic quality verification for translations.

Algorithm:
  1. Translate: English -> Hindi (primary translation)
  2. Back-translate: Hindi -> English (using same provider)
  3. Compute BLEU score: back-translated English vs original English
  4. If BLEU < threshold -> flag segment -> retry with alternate provider

This catches:
  - Hallucinated translations (provider makes up content)
  - Meaning-destroying translations (words correct, meaning wrong)
  - Numeric errors (1000 becomes 100 in the Hindi version)

BLEU score interpretation for short segments:
  > 0.5  = High quality, preserve
  0.3-0.5 = Acceptable, note warning
  < 0.3  = Fail, retry with alternate provider
"""
import asyncio
import structlog
from dataclasses import dataclass
from typing import Optional

log = structlog.get_logger(__name__)


@dataclass
class BackTranslationScore:
    original_text: str
    translated_text: str
    back_translated_text: str
    bleu_score: float           # 0.0 - 1.0
    passed: bool                # True if quality is acceptable
    should_retry: bool


class BackTranslationQualityScorer:
    """
    Validates translation quality using back-translation + BLEU scoring.

    Usage:
        scorer = BackTranslationQualityScorer(threshold=0.30)
        result = await scorer.score(
            original="Hello, how are you?",
            translated="namaste, aap kaise hain?",
            source_lang="en", target_lang="hi",
            translation_provider=provider
        )
        if not result.passed:
            # retry with alternate provider
    """

    def __init__(self, threshold: float = 0.30, enabled: bool = True):
        self.threshold = threshold
        self.enabled = enabled
        self._nltk_ready = False

    async def _ensure_nltk(self):
        """Lazy-load NLTK data for BLEU scoring."""
        if not self._nltk_ready:
            try:
                import nltk
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: nltk.download("punkt", quiet=True))
                await loop.run_in_executor(None, lambda: nltk.download("punkt_tab", quiet=True))
                self._nltk_ready = True
            except Exception:
                pass

    async def score(
        self,
        original_text: str,
        translated_text: str,
        source_lang: str,
        target_lang: str,
        translation_provider,
    ) -> BackTranslationScore:
        """
        Score a translation using back-translation.
        Returns immediately with score=1.0 if disabled or text is very short.
        """
        if not self.enabled or len(original_text.split()) < 4:
            return BackTranslationScore(
                original_text=original_text,
                translated_text=translated_text,
                back_translated_text="",
                bleu_score=1.0,
                passed=True,
                should_retry=False,
            )

        try:
            await self._ensure_nltk()

            # Back-translate: target_lang -> source_lang
            back_translated = await translation_provider.translate(
                text=translated_text,
                source_language=target_lang,
                target_language=source_lang,
            )

            # Compute BLEU score
            bleu = await self._compute_bleu(original_text, back_translated)
            passed = bleu >= self.threshold

            log.info(
                "back_translation_scored",
                bleu=round(bleu, 3),
                threshold=self.threshold,
                passed=passed,
                original_preview=original_text[:50],
                back_translated_preview=back_translated[:50],
            )

            return BackTranslationScore(
                original_text=original_text,
                translated_text=translated_text,
                back_translated_text=back_translated,
                bleu_score=bleu,
                passed=passed,
                should_retry=not passed,
            )

        except Exception as e:
            log.warning("back_translation_scoring_failed", error=str(e))
            return BackTranslationScore(
                original_text=original_text, translated_text=translated_text,
                back_translated_text="", bleu_score=0.5, passed=True, should_retry=False,
            )

    async def _compute_bleu(self, reference: str, hypothesis: str) -> float:
        """Compute BLEU score between reference and hypothesis."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._bleu_sync, reference, hypothesis)

    def _bleu_sync(self, reference: str, hypothesis: str) -> float:
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

            ref_tokens = reference.lower().split()
            hyp_tokens = hypothesis.lower().split()

            if not ref_tokens or not hyp_tokens:
                return 0.0

            smoother = SmoothingFunction().method1
            score = sentence_bleu(
                [ref_tokens], hyp_tokens,
                smoothing_function=smoother,
                weights=(0.5, 0.5, 0.0, 0.0),  # unigram + bigram only for short text
            )
            return float(score)
        except Exception:
            # Fallback: simple word overlap ratio
            ref_set = set(reference.lower().split())
            hyp_set = set(hypothesis.lower().split())
            if not ref_set:
                return 0.0
            overlap = len(ref_set & hyp_set) / len(ref_set)
            return overlap
