"""
Multi-Tier NLP Sentiment Analysis Engine.

Combines:
- Tier 1: Loughran-McDonald financial dictionary & VADER-style rule-based engine
  with financial phrase matching, valence shifters, intensifiers, and French financial lexicons.
- Tier 2: FinBERT / HuggingFace Transformer deep neural sentiment model with automatic batching.
- Resilient fallback mechanism: Guaranteed high-throughput zero-latency fallback to Tier 1.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from src.sentiment.rss_crawler import NewsArticle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier 1: Financial Lexicon & Domain Knowledge Base (Loughran-McDonald + Phrases)
# ---------------------------------------------------------------------------

FINANCIAL_POSITIVE_WORDS: Dict[str, float] = {
    # Loughran-McDonald & Financial Markets Positive Lexicon
    "beat": 1.4,
    "beats": 1.4,
    "beating": 1.4,
    "bull": 1.2,
    "bullish": 1.4,
    "surge": 1.3,
    "surged": 1.3,
    "surging": 1.3,
    "soar": 1.4,
    "soared": 1.4,
    "soaring": 1.4,
    "gain": 1.0,
    "gained": 1.0,
    "gains": 1.0,
    "growth": 1.1,
    "grow": 1.0,
    "growing": 1.0,
    "grew": 1.0,
    "profit": 1.2,
    "profitable": 1.3,
    "profitability": 1.4,
    "profits": 1.2,
    "outperform": 1.5,
    "outperformed": 1.5,
    "outperforming": 1.5,
    "outperformance": 1.5,
    "upgrade": 1.4,
    "upgraded": 1.4,
    "upgrades": 1.4,
    "upgrading": 1.4,
    "record": 1.2,
    "rebound": 1.2,
    "rebounded": 1.2,
    "rebounding": 1.2,
    "rally": 1.3,
    "rallied": 1.3,
    "rallying": 1.3,
    "dividend": 1.0,
    "dividends": 1.0,
    "buyback": 1.2,
    "buybacks": 1.2,
    "breakthrough": 1.4,
    "lucrative": 1.3,
    "exceed": 1.3,
    "exceeded": 1.3,
    "exceeding": 1.3,
    "surpass": 1.3,
    "surpassed": 1.3,
    "surpassing": 1.3,
    "upside": 1.2,
    "optimism": 1.1,
    "optimistic": 1.2,
    "strong": 1.1,
    "stronger": 1.2,
    "strongest": 1.3,
    "strength": 1.1,
    "expansion": 1.1,
    "expanded": 1.1,
    "expanding": 1.1,
    "innovative": 1.1,
    "innovation": 1.1,
    "synergy": 1.1,
    "synergies": 1.2,
    "resilient": 1.2,
    "resilience": 1.2,
    "boom": 1.3,
    "booming": 1.3,
    "success": 1.2,
    "successful": 1.2,
    "achieve": 1.1,
    "achieved": 1.1,
    "achievement": 1.2,
    "triumph": 1.3,
    "favorable": 1.1,
    "accelerate": 1.1,
    "accelerated": 1.1,
    "acceleration": 1.2,
    "leader": 1.0,
    "leadership": 1.1,
    "windfall": 1.4,
    "tailwind": 1.2,
    "tailwinds": 1.3,
    "milestone": 1.1,
    "monetize": 1.1,
    "monetization": 1.1,
    "partnership": 1.0,
    "alliance": 1.0,
    "unveil": 0.8,
    "unveiled": 0.8,
    "unveils": 0.8,
    "boost": 1.2,
    "boosted": 1.2,
    "boosting": 1.2,
    "jump": 1.1,
    "jumped": 1.1,
    "jumping": 1.1,
    "climb": 1.0,
    "climbed": 1.0,
    "climbing": 1.0,
    "upward": 1.0,
    "upswing": 1.2,
    "uptrend": 1.3,
    "solid": 1.0,
    "robust": 1.2,
    "stellar": 1.5,
    "exceptional": 1.4,
    "premium": 0.9,
    # French Positive Terms (Moroccan Equities)
    "hausse": 1.2,
    "croissance": 1.2,
    "progression": 1.1,
    "bénéfice": 1.3,
    "bénéfices": 1.3,
    "surperformance": 1.5,
    "surperformer": 1.5,
    "dividende": 1.0,
    "dividendes": 1.0,
    "gain": 1.0,
    "gains": 1.0,
    "record": 1.2,
    "rebond": 1.2,
    "rentable": 1.3,
    "rentabilité": 1.4,
    "embellie": 1.2,
    "optimiste": 1.2,
    "optimisme": 1.1,
    "dynamique": 1.0,
    "consolidation": 0.8,
    "renforcement": 1.0,
    "expansion": 1.1,
    "succès": 1.2,
    "solide": 1.0,
    "solides": 1.0,
    "robuste": 1.2,
    "bond": 1.2,
    "envolée": 1.3,
    "revalorisation": 1.2,
    "surévaluation": 0.8,
}

FINANCIAL_NEGATIVE_WORDS: Dict[str, float] = {
    # Loughran-McDonald & Financial Markets Negative Lexicon
    "miss": 1.4,
    "missed": 1.4,
    "missing": 1.3,
    "misses": 1.4,
    "drop": 1.2,
    "dropped": 1.2,
    "dropping": 1.2,
    "drops": 1.2,
    "fall": 1.1,
    "fell": 1.1,
    "falling": 1.1,
    "falls": 1.1,
    "plunge": 1.4,
    "plunged": 1.4,
    "plunging": 1.4,
    "plunges": 1.4,
    "slump": 1.3,
    "slumped": 1.3,
    "slumping": 1.3,
    "slumps": 1.3,
    "tumble": 1.3,
    "tumbled": 1.3,
    "tumbling": 1.3,
    "tumbles": 1.3,
    "crash": 1.5,
    "crashed": 1.5,
    "crashing": 1.5,
    "loss": 1.3,
    "losses": 1.3,
    "lost": 1.1,
    "losing": 1.1,
    "decline": 1.2,
    "declined": 1.2,
    "declines": 1.2,
    "declining": 1.2,
    "downgrade": 1.5,
    "downgraded": 1.5,
    "downgrades": 1.5,
    "downgrading": 1.5,
    "underperform": 1.4,
    "underperformed": 1.4,
    "underperforming": 1.4,
    "underperformance": 1.4,
    "deficit": 1.3,
    "deficits": 1.3,
    "debt": 1.1,
    "debts": 1.1,
    "bankrupt": 1.8,
    "bankruptcy": 1.8,
    "insolvent": 1.7,
    "insolvency": 1.7,
    "default": 1.6,
    "defaulted": 1.6,
    "defaulting": 1.6,
    "lawsuit": 1.4,
    "lawsuits": 1.4,
    "sue": 1.3,
    "sued": 1.3,
    "suing": 1.3,
    "litigation": 1.4,
    "litigious": 1.3,
    "fraud": 1.8,
    "fraudulent": 1.8,
    "investigation": 1.4,
    "investigating": 1.4,
    "subpoena": 1.5,
    "subpoenas": 1.5,
    "penalty": 1.4,
    "penalties": 1.4,
    "fine": 1.2,
    "fined": 1.3,
    "fines": 1.2,
    "sanction": 1.4,
    "sanctions": 1.4,
    "warning": 1.3,
    "warned": 1.3,
    "warns": 1.3,
    "cut": 1.2,
    "cuts": 1.2,
    "cutting": 1.2,
    "layoff": 1.3,
    "layoffs": 1.3,
    "fired": 1.2,
    "restructure": 1.1,
    "restructuring": 1.1,
    "weak": 1.2,
    "weaker": 1.2,
    "weakest": 1.3,
    "weakness": 1.3,
    "struggle": 1.2,
    "struggling": 1.2,
    "struggles": 1.2,
    "struggled": 1.2,
    "deteriorate": 1.4,
    "deteriorated": 1.4,
    "deterioration": 1.4,
    "headwind": 1.2,
    "headwinds": 1.3,
    "inflation": 1.0,
    "bear": 1.1,
    "bearish": 1.4,
    "selloff": 1.4,
    "sell": 1.0,
    "selling": 1.0,
    "downtrend": 1.3,
    "downside": 1.2,
    "recession": 1.4,
    "recessionary": 1.4,
    "scandal": 1.6,
    "crisis": 1.5,
    "panic": 1.5,
    "dismal": 1.4,
    "disappointing": 1.3,
    "disappointment": 1.3,
    "halt": 1.3,
    "halted": 1.3,
    "suspension": 1.4,
    "suspended": 1.4,
    "delay": 1.1,
    "delayed": 1.1,
    "delays": 1.1,
    "impairment": 1.4,
    "writeoff": 1.5,
    "writedown": 1.5,
    "bleak": 1.3,
    # French Negative Terms (Moroccan Equities)
    "baisse": 1.2,
    "recul": 1.2,
    "repli": 1.1,
    "chute": 1.4,
    "perte": 1.3,
    "pertes": 1.3,
    "déficit": 1.3,
    "déficits": 1.3,
    "dégradation": 1.4,
    "dégradé": 1.4,
    "sousperformance": 1.4,
    "litige": 1.4,
    "litiges": 1.4,
    "sanction": 1.4,
    "sanctions": 1.4,
    "sanctionné": 1.4,
    "fraude": 1.8,
    "faillite": 1.8,
    "avertissement": 1.3,
    "ralentissement": 1.2,
    "crise": 1.5,
    "dette": 1.1,
    "dettes": 1.1,
    "défaillance": 1.6,
    "retard": 1.1,
    "retards": 1.1,
    "inflation": 1.0,
    "pessimiste": 1.2,
    "pessimisme": 1.2,
    "contraction": 1.3,
}

FINANCIAL_PHRASES: Dict[str, float] = {
    # High-conviction multi-word financial phrases (positive)
    "beat estimates": 1.8,
    "beats estimates": 1.8,
    "beating estimates": 1.8,
    "beat earnings": 1.8,
    "beats earnings": 1.8,
    "guidance raised": 1.9,
    "raises guidance": 1.9,
    "raised guidance": 1.9,
    "raising guidance": 1.9,
    "all-time high": 1.6,
    "all time high": 1.6,
    "record high": 1.6,
    "record profit": 1.7,
    "record earnings": 1.7,
    "record revenue": 1.7,
    "revenue growth": 1.4,
    "profit growth": 1.5,
    "dividend hike": 1.5,
    "dividend increase": 1.5,
    "boosts dividend": 1.5,
    "share buyback": 1.4,
    "stock buyback": 1.4,
    "repurchase program": 1.3,
    "upgraded to buy": 1.8,
    "upgrade to buy": 1.8,
    "upgraded to outperform": 1.8,
    "margin expansion": 1.6,
    "positive outlook": 1.5,
    "earnings beat": 1.8,
    "earnings surprise": 1.7,
    "strategic partnership": 1.3,
    "market share gains": 1.4,
    "fda approval": 1.8,
    "favorable ruling": 1.5,
    "strong demand": 1.4,
    "strong cash flow": 1.5,
    "free cash flow surged": 1.7,
    "bénéfice net en hausse": 1.8,
    "résultat net en hausse": 1.8,
    "chiffre d'affaires en hausse": 1.6,
    "hausse du dividende": 1.5,
    "forte croissance": 1.6,
    # High-conviction multi-word financial phrases (negative)
    "missed estimates": -1.8,
    "misses estimates": -1.8,
    "missing estimates": -1.8,
    "missed earnings": -1.8,
    "misses earnings": -1.8,
    "guidance lowered": -1.9,
    "lowers guidance": -1.9,
    "lowered guidance": -1.9,
    "cuts guidance": -1.9,
    "earnings miss": -1.8,
    "revenue drop": -1.5,
    "profit plunge": -1.7,
    "dividend cut": -1.7,
    "cuts dividend": -1.7,
    "slashes dividend": -1.8,
    "downgraded to sell": -1.8,
    "downgrade to sell": -1.8,
    "downgraded to underperform": -1.8,
    "sec investigation": -2.0,
    "accounting fraud": -2.0,
    "lawsuit filed": -1.5,
    "class action lawsuit": -1.6,
    "bankruptcy filing": -2.0,
    "filed for bankruptcy": -2.0,
    "chapter 11": -2.0,
    "supply chain disruption": -1.4,
    "margin contraction": -1.6,
    "regulatory fine": -1.5,
    "profit warning": -1.8,
    "warning on profits": -1.8,
    "credit downgrade": -1.7,
    "debt default": -1.9,
    "product recall": -1.5,
    "weak guidance": -1.6,
    "lowers outlook": -1.6,
    "disappointing earnings": -1.6,
    "bénéfice net en baisse": -1.8,
    "résultat net en baisse": -1.8,
    "chiffre d'affaires en baisse": -1.6,
    "baisse du dividende": -1.7,
    "perte nette": -1.8,
    "avertissement sur résultats": -1.8,
}

NEGATION_WORDS: Set[str] = {
    "not",
    "no",
    "never",
    "neither",
    "nobody",
    "none",
    "nor",
    "nothing",
    "nowhere",
    "hardly",
    "scarcely",
    "barely",
    "doesnt",
    "doesn't",
    "isnt",
    "isn't",
    "wasnt",
    "wasn't",
    "shouldnt",
    "shouldn't",
    "wouldnt",
    "wouldn't",
    "couldnt",
    "couldn't",
    "wont",
    "won't",
    "cant",
    "can't",
    "cannot",
    "without",
    "against",
    "fails",
    "failed",
    "failing",
    # French Negations
    "pas",
    "sans",
    "non",
    "aucun",
    "aucune",
    "jamais",
    "rien",
    "ne",
    "n'",
}

INTENSIFIER_WORDS: Dict[str, float] = {
    "significantly": 1.4,
    "massively": 1.5,
    "strongly": 1.4,
    "hugely": 1.4,
    "substantially": 1.3,
    "extremely": 1.4,
    "exceptionally": 1.5,
    "dramatically": 1.4,
    "sharp": 1.3,
    "sharply": 1.3,
    "steep": 1.3,
    "steeply": 1.3,
    "fortement": 1.4,
    "nettement": 1.4,
    "drastiquement": 1.4,
    "très": 1.3,
    # Dampeners (< 1.0)
    "slightly": 0.6,
    "marginally": 0.6,
    "moderately": 0.7,
    "partially": 0.7,
    "légèrement": 0.6,
    "modérément": 0.7,
    "faiblement": 0.6,
}


@dataclass
class SentimentResult:
    """Standardized Sentiment NLP Output Record."""

    article_id: str
    symbol: str
    published_at_utc: datetime
    sentiment_score: float  # [-1.0, 1.0]
    sentiment_confidence: float  # [0.0, 1.0]
    sentiment_label: str  # 'POSITIVE' | 'NEUTRAL' | 'NEGATIVE'
    probabilities: Dict[str, float] = field(default_factory=dict)
    model_name: str = "financial_lexicon_tier1"
    inference_latency_ms: float = 0.0
    processed_at_utc: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> Dict[str, object]:
        return {
            "article_id": self.article_id,
            "symbol": self.symbol,
            "published_at_utc": self.published_at_utc.isoformat(),
            "sentiment_score": self.sentiment_score,
            "sentiment_confidence": self.sentiment_confidence,
            "sentiment_label": self.sentiment_label,
            "probabilities": self.probabilities,
            "model_name": self.model_name,
            "inference_latency_ms": self.inference_latency_ms,
            "processed_at_utc": self.processed_at_utc.isoformat(),
        }


class FinancialLexiconScorer:
    """
    Tier 1 Financial Lexicon & Rule-Based Scorer.
    Zero latency (<0.1ms), zero external model dependency, deterministic and robust.
    """

    def __init__(self, phrase_weight: float = 1.5, title_weight: float = 1.6) -> None:
        self.phrase_weight = phrase_weight
        self.title_weight = title_weight

    def score_text(
        self, text: str, title: Optional[str] = None
    ) -> Tuple[float, float, str, Dict[str, float]]:
        """
        Score input text using financial lexicon rules, phrases, negations, and intensifiers.
        Returns (sentiment_score, confidence, label, probabilities).
        """
        full_text = f"{title or ''} {text}".strip().lower()
        if not full_text:
            return (
                0.0,
                0.0,
                "NEUTRAL",
                {"positive": 0.333, "neutral": 0.334, "negative": 0.333},
            )

        # 1. Evaluate Multi-Word High-Conviction Financial Phrases
        phrase_pos_score = 0.0
        phrase_neg_score = 0.0

        for phrase, weight in FINANCIAL_PHRASES.items():
            if phrase in full_text:
                count = full_text.count(phrase)
                if weight > 0:
                    phrase_pos_score += weight * count * self.phrase_weight
                else:
                    phrase_neg_score += abs(weight) * count * self.phrase_weight

        # 2. Token-Level Lexicon Evaluation with Negations & Intensifiers
        tokens = re.findall(
            r"\b[a-zA-Z0-9_\u00C0-\u017F\u0600-\u06FF]{2,}\b", full_text
        )
        if not tokens and phrase_pos_score == 0 and phrase_neg_score == 0:
            return (
                0.0,
                0.0,
                "NEUTRAL",
                {"positive": 0.333, "neutral": 0.334, "negative": 0.333},
            )

        token_pos_score = 0.0
        token_neg_score = 0.0
        total_valence_words = 0

        n_tokens = len(tokens)
        for i, token in enumerate(tokens):
            # Check if preceded by negation within 3 tokens
            is_negated = False
            lookback_start = max(0, i - 3)
            for prev_token in tokens[lookback_start:i]:
                if prev_token in NEGATION_WORDS:
                    is_negated = True
                    break

            # Check if preceded by intensifier/dampener
            intensity_multiplier = 1.0
            if i > 0 and tokens[i - 1] in INTENSIFIER_WORDS:
                intensity_multiplier = INTENSIFIER_WORDS[tokens[i - 1]]

            # Match positive dictionary
            if token in FINANCIAL_POSITIVE_WORDS:
                total_valence_words += 1
                raw_val = FINANCIAL_POSITIVE_WORDS[token] * intensity_multiplier
                if is_negated:
                    # Invert into negative valence
                    token_neg_score += raw_val * 0.9
                else:
                    token_pos_score += raw_val

            # Match negative dictionary
            elif token in FINANCIAL_NEGATIVE_WORDS:
                total_valence_words += 1
                raw_val = FINANCIAL_NEGATIVE_WORDS[token] * intensity_multiplier
                if is_negated:
                    # Negation of negative word (e.g. "not default", "not fail")
                    token_pos_score += raw_val * 0.7
                else:
                    token_neg_score += raw_val

        # Combine phrase and token scores
        total_pos = token_pos_score + phrase_pos_score
        total_neg = token_neg_score + phrase_neg_score

        # Title boost if title was provided separately
        if title:
            title_clean = title.strip().lower()
            for phrase, weight in FINANCIAL_PHRASES.items():
                if phrase in title_clean:
                    if weight > 0:
                        total_pos += weight * (self.title_weight - 1.0)
                    else:
                        total_neg += abs(weight) * (self.title_weight - 1.0)

        diff = total_pos - total_neg
        total_mass = total_pos + total_neg

        if total_mass == 0:
            return (
                0.0,
                0.0,
                "NEUTRAL",
                {"positive": 0.333, "neutral": 0.334, "negative": 0.333},
            )

        # Continuous score S in [-1.0, 1.0] using smoothed hyperbolic tangent
        # Normalized by square root of length to prevent dilution on long articles
        scale = max(1.0, math.sqrt(len(tokens) + 1) * 0.75)
        raw_ratio = diff / scale
        sentiment_score = float(math.tanh(raw_ratio))

        # Clamp strictly to [-1.0, 1.0]
        sentiment_score = max(-1.0, min(1.0, round(sentiment_score, 4)))

        # Compute confidence in [0.0, 1.0]
        # Confidence increases with net polarized mass and signal consistency
        dominance = abs(diff) / (total_mass + 1e-6)
        mass_factor = min(1.0, total_mass / 3.0)
        confidence = float(min(1.0, max(0.1, 0.4 * dominance + 0.6 * mass_factor)))
        confidence = round(confidence, 4)

        # Derive probability distribution
        pos_signal = max(0.0, sentiment_score)
        neg_signal = max(0.0, -sentiment_score)

        p_pos = pos_signal * confidence + (1.0 - confidence) * 0.333
        p_neg = neg_signal * confidence + (1.0 - confidence) * 0.333
        p_neu = max(0.0, 1.0 - (p_pos + p_neg))

        sum_p = p_pos + p_neg + p_neu
        p_pos /= sum_p
        p_neg /= sum_p
        p_neu /= sum_p

        probabilities = {
            "positive": round(p_pos, 4),
            "neutral": round(p_neu, 4),
            "negative": round(p_neg, 4),
        }

        # Determine discrete label
        if sentiment_score >= 0.15:
            label = "POSITIVE"
        elif sentiment_score <= -0.15:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        return sentiment_score, confidence, label, probabilities


class TransformerSentimentScorer:
    """
    Tier 2 Deep Transformer NLP Scorer (FinBERT via HuggingFace & PyTorch).
    Provides deep contextual sentiment inference with automatic batching.
    """

    def __init__(
        self,
        model_name_or_path: str = "ProsusAI/finbert",
        device: Optional[str] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self._device = device
        self._tokenizer = None
        self._model = None
        self._is_loaded = False
        self._load_attempted = False

    def is_available(self) -> bool:
        """Check whether transformer model and PyTorch dependencies are available."""
        if not self._load_attempted:
            self._try_load()
        return self._is_loaded

    def _try_load(self) -> None:
        self._load_attempted = True
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(
                "Loading FinBERT model from '%s' on %s...",
                self.model_name_or_path,
                self._device,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, local_files_only=False
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path, local_files_only=False
            )
            self._model.to(self._device)
            self._model.eval()
            self._is_loaded = True
            logger.info("FinBERT model successfully initialized.")
        except Exception as e:
            logger.info(
                "Transformer model '%s' unavailable (offline/not cached): %s. Tier 1 fallback active.",
                self.model_name_or_path,
                e,
            )
            self._is_loaded = False

    def score_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Tuple[float, float, str, Dict[str, float]]]:
        """Score a list of text strings in mini-batches."""
        if not self.is_available():
            raise RuntimeError("Transformer model is not loaded.")

        import torch

        results: List[Tuple[float, float, str, Dict[str, float]]] = []

        # FinBERT labels mapping: usually 0: positive, 1: negative, 2: neutral (or id2label)
        id2label = getattr(self._model.config, "id2label", None) or {
            0: "positive",
            1: "negative",
            2: "neutral",
        }
        # Normalize label mapping strings to lowercase
        id2label = {idx: label.lower() for idx, label in id2label.items()}

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            encoded = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}

            with torch.no_grad():
                logits = self._model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for p_row in probs:
                p_dict = {id2label.get(idx, f"label_{idx}"): float(p_row[idx]) for idx in range(len(p_row))}
                p_pos = p_dict.get("positive", 0.0)
                p_neg = p_dict.get("negative", 0.0)
                p_neu = p_dict.get("neutral", 0.0)

                score = float(p_pos - p_neg)
                score = max(-1.0, min(1.0, round(score, 4)))
                confidence = float(max(p_pos, p_neg, p_neu))
                confidence = max(0.0, min(1.0, round(confidence, 4)))

                if score >= 0.15:
                    label = "POSITIVE"
                elif score <= -0.15:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"

                prob_summary = {
                    "positive": round(p_pos, 4),
                    "neutral": round(p_neu, 4),
                    "negative": round(p_neg, 4),
                }
                results.append((score, confidence, label, prob_summary))

        return results


class NLPSentimentEngine:
    """
    Unified Multi-Tier NLP Sentiment Engine.
    Routes inference to Tier 2 (FinBERT) when available; otherwise uses Tier 1 (Lexicon).
    """

    def __init__(
        self,
        prefer_transformer: bool = True,
        transformer_model: str = "ProsusAI/finbert",
        device: Optional[str] = None,
    ) -> None:
        self.prefer_transformer = prefer_transformer
        self.lexicon_scorer = FinancialLexiconScorer()
        self.transformer_scorer = (
            TransformerSentimentScorer(model_name_or_path=transformer_model, device=device)
            if prefer_transformer
            else None
        )

    def score_text(
        self, text: str, title: Optional[str] = None
    ) -> Tuple[float, float, str, Dict[str, float]]:
        """Score text directly, returning (sentiment_score, confidence, label, probabilities)."""
        combined = f"{title or ''}. {text}".strip()
        if self.prefer_transformer and self.transformer_scorer and self.transformer_scorer.is_available():
            try:
                res = self.transformer_scorer.score_batch([combined])[0]
                return res
            except Exception as e:
                logger.warning("Transformer scoring failed: %s, falling back to Lexicon.", e)

        return self.lexicon_scorer.score_text(text, title=title)

    def score_article(self, article: NewsArticle) -> SentimentResult:
        """Score a single NewsArticle instance."""
        t0 = time.perf_counter()
        score, conf, label, probs = self.score_text(
            text=article.summary or article.title,
            title=article.title,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        model_name = (
            "finbert_transformer"
            if self.prefer_transformer
            and self.transformer_scorer
            and self.transformer_scorer.is_available()
            else "financial_lexicon_tier1"
        )

        return SentimentResult(
            article_id=article.article_id,
            symbol=article.symbol,
            published_at_utc=article.published_at_utc,
            sentiment_score=score,
            sentiment_confidence=conf,
            sentiment_label=label,
            probabilities=probs,
            model_name=model_name,
            inference_latency_ms=round(latency_ms, 2),
            processed_at_utc=datetime.now(timezone.utc),
        )

    def score_batch(
        self, articles: Sequence[NewsArticle], batch_size: int = 32
    ) -> List[SentimentResult]:
        """Score a sequence of NewsArticle instances efficiently in batches."""
        if not articles:
            return []

        results: List[SentimentResult] = []
        t0 = time.perf_counter()

        # Check if transformer model is available for batch inference
        if (
            self.prefer_transformer
            and self.transformer_scorer
            and self.transformer_scorer.is_available()
        ):
            try:
                texts = [f"{art.title}. {art.summary}".strip() for art in articles]
                scored = self.transformer_scorer.score_batch(texts, batch_size=batch_size)
                total_latency_ms = (time.perf_counter() - t0) * 1000.0
                per_item_latency = total_latency_ms / len(articles)

                for art, (score, conf, label, probs) in zip(articles, scored):
                    results.append(
                        SentimentResult(
                            article_id=art.article_id,
                            symbol=art.symbol,
                            published_at_utc=art.published_at_utc,
                            sentiment_score=score,
                            sentiment_confidence=conf,
                            sentiment_label=label,
                            probabilities=probs,
                            model_name="finbert_transformer",
                            inference_latency_ms=round(per_item_latency, 2),
                            processed_at_utc=datetime.now(timezone.utc),
                        )
                    )
                return results
            except Exception as e:
                logger.warning(
                    "Transformer batch scoring failed: %s, falling back to Lexicon.", e
                )

        # Tier 1 Lexicon batch inference
        for art in articles:
            results.append(self.score_article(art))

        return results
