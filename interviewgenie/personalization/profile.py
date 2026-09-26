"""Personalisation: learn the interviewee and adapt the answers to them.

The :class:`IntervieweeProfile` is a small, interpretable preference model:

* **static facts** (name, target role, seniority, strengths, story bank),
* **style preferences** learned from explicit feedback (formality, verbosity,
  hedging, use of numbers, humour) with exponential moving averages,
* **evidence weights** for the kinds of support the candidate prefers
  (metrics, technical depth, people stories), and
* **topic affinities** used to pick which of several valid angles to lead with.

:class:`StyleAdapter` turns that profile into concrete generator parameters.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..logging import get_logger
from ..types import Analysis

LOG = get_logger("personalization.profile")

#: style dimensions the model tracks, with (default, min, max)
STYLE_DIMENSIONS: Dict[str, Tuple[float, float, float]] = {
    "formality": (0.65, 0.0, 1.0),
    "verbosity": (0.55, 0.0, 1.0),
    "hedging": (0.30, 0.0, 1.0),
    "concreteness": (0.70, 0.0, 1.0),
    "enthusiasm": (0.60, 0.0, 1.0),
    "technical_depth": (0.60, 0.0, 1.0),
    "people_focus": (0.45, 0.0, 1.0),
}

#: how strongly a piece of feedback moves each dimension
FEEDBACK_SENSITIVITY: Dict[str, Dict[str, float]] = {
    "too long": {"verbosity": -0.25, "concreteness": -0.05},
    "too short": {"verbosity": 0.25, "concreteness": 0.05},
    "too formal": {"formality": -0.25},
    "too casual": {"formality": 0.25},
    "too vague": {"concreteness": -0.25, "hedging": -0.05},
    "too technical": {"technical_depth": -0.25, "formality": -0.05},
    "not technical enough": {"technical_depth": 0.25},
    "too much detail": {"verbosity": -0.2, "technical_depth": -0.1},
    "more numbers": {"concreteness": 0.25},
    "less hedging": {"hedging": -0.25},
    "more enthusiasm": {"enthusiasm": 0.25},
    "calmer": {"enthusiasm": -0.25},
    "more about the team": {"people_focus": 0.25},
    "less about me": {"people_focus": -0.25},
    "perfect": {},
    "good": {},
    "great": {},
    "that worked": {},
}


@dataclass
class IntervieweeProfile:
    """Everything the system has learned about the person being coached."""

    name: str = "Candidate"
    target_role: str = ""
    target_company: str = ""
    seniority: str = ""
    years_experience: Optional[int] = None
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    story_bank: List[Dict[str, str]] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    #: Free-text context the candidate supplies before the interview: the job
    #: description they are interviewing for and/or their résumé.  Both are fed
    #: into the generation prompt so answers reference the real role and the
    #: candidate's real experience rather than generic filler.
    job_description: str = ""
    resume_text: str = ""
    style: Dict[str, float] = field(
        default_factory=lambda: {k: v[0] for k, v in STYLE_DIMENSIONS.items()})
    topic_affinity: Dict[str, float] = field(default_factory=dict)
    evidence_weights: Dict[str, float] = field(default_factory=lambda: {
        "metrics": 0.6, "technical_depth": 0.6, "people": 0.5, "process": 0.5})
    feedback_log: List[Dict[str, Any]] = field(default_factory=list)
    interactions: int = 0
    learning_rate: float = 0.25

    # -- static facts ----------------------------------------------------- #
    def update_facts(self, **facts: Any) -> None:
        for key, value in facts.items():
            if value in (None, "", [], {}):
                continue
            if isinstance(value, list):
                current = getattr(self, key, None)
                if isinstance(current, list):
                    # case-insensitive merge: "Kafka" and "kafka" are one skill,
                    # and a duplicated list makes the prompt look sloppy
                    seen = {str(existing).strip().lower() for existing in current}
                    for item in value:
                        marker = str(item).strip().lower()
                        if marker and marker not in seen:
                            current.append(item)
                            seen.add(marker)
                    continue
            setattr(self, key, value)

    def add_story(self, title: str, situation: str, task: str, action: str,
                  result: str, tags: Optional[Sequence[str]] = None,
                  learning: str = "") -> None:
        self.story_bank.append({
            "title": title, "situation": situation, "task": task,
            "action": action, "result": result, "learning": learning,
            "tags": list(tags or []),
        })

    def stories_for(self, topic: str, limit: int = 2) -> List[Dict[str, str]]:
        topic_low = topic.lower()
        scored = []
        for story in self.story_bank:
            text = " ".join(str(v) for v in story.values() if isinstance(v, str)).lower()
            score = sum(1 for tag in story.get("tags", []) if str(tag).lower() in topic_low)
            score += 2 if topic_low in text else 0
            if score > 0:
                scored.append((score, story))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [story for _score, story in scored[:limit]]

    # -- learning --------------------------------------------------------- #
    def apply_feedback(self, feedback: str, learning_rate: Optional[float] = None) -> Dict[str, float]:
        """Move the style vector in response to natural-language feedback."""
        rate = learning_rate if learning_rate is not None else self.learning_rate
        low = feedback.lower().strip()
        changes: Dict[str, float] = {}
        for phrase, deltas in FEEDBACK_SENSITIVITY.items():
            if phrase in low:
                for dimension, delta in deltas.items():
                    current = self.style.get(dimension, STYLE_DIMENSIONS[dimension][0])
                    lo, hi = STYLE_DIMENSIONS[dimension][1], STYLE_DIMENSIONS[dimension][2]
                    updated = max(lo, min(hi, current + delta * rate * 2))
                    self.style[dimension] = round(updated, 4)
                    changes[dimension] = round(updated - current, 4)
        self.feedback_log.append({"feedback": feedback, "changes": changes})
        if len(self.feedback_log) > 200:
            self.feedback_log = self.feedback_log[-200:]
        return changes

    def observe_answer(self, text: str, reward: float = 0.5) -> None:
        """Nudge style dimensions towards what actually worked (implicit feedback)."""
        self.interactions += 1
        low = text.lower()
        words = len(text.split())
        if reward >= 0.6:
            self.style["verbosity"] = _move(self.style["verbosity"],
                                            min(1.0, words / 110), 0.08)
            self.style["concreteness"] = _move(
                self.style["concreteness"],
                1.0 if any(c.isdigit() for c in text) else 0.4, 0.08)
            self.style["enthusiasm"] = _move(
                self.style["enthusiasm"], 1.0 if "!" in text else 0.5, 0.05)

    def note_topic(self, topic: str, weight: float = 1.0) -> None:
        self.topic_affinity[topic] = self.topic_affinity.get(topic, 0.0) + weight

    # -- personalisation score -------------------------------------------- #
    def personalisation_score(self, text: str) -> float:
        """How much ``text`` reflects this candidate's profile and style."""
        low = text.lower()
        score = 0.0
        if self.name and self.name != "Candidate" and self.name.lower() in low:
            score += 0.15
        if self.target_role and self.target_role.lower() in low:
            score += 0.15
        if self.target_company and self.target_company.lower() in low:
            score += 0.1
        for skill in self.skills[:6]:
            if skill.lower() in low:
                score += 0.08
        for strength in self.strengths[:4]:
            if strength.lower() in low:
                score += 0.08
        for goal in self.goals[:3]:
            if goal.lower() in low:
                score += 0.06
        # style fit: verbosity alignment
        words = len(text.split())
        target_words = 30 + self.style.get("verbosity", 0.55) * 90
        closeness = 1.0 - min(1.0, abs(words - target_words) / max(30.0, target_words))
        score += 0.2 * closeness
        # hedging alignment
        hedges = sum(low.count(h) for h in ("maybe", "i think", "perhaps", "probably", "sort of"))
        expected = self.style.get("hedging", 0.3) * 4
        if expected > 0.5:
            score += 0.1 * min(1.0, hedges / max(1.0, expected))
        else:
            score += 0.1 if hedges == 0 else 0.0
        return round(max(0.0, min(1.0, score)), 4)

    # -- persistence ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "target_role": self.target_role,
            "target_company": self.target_company, "seniority": self.seniority,
            "years_experience": self.years_experience,
            "strengths": self.strengths, "weaknesses": self.weaknesses,
            "story_bank": self.story_bank, "skills": self.skills, "goals": self.goals,
            "constraints": self.constraints,
            "job_description": self.job_description,
            "resume_text": self.resume_text,
            "style": self.style,
            "topic_affinity": self.topic_affinity,
            "evidence_weights": self.evidence_weights,
            "interactions": self.interactions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntervieweeProfile":
        profile = cls()
        profile.update_facts(**{k: v for k, v in data.items()
                                if k in {"name", "target_role", "target_company",
                                         "seniority", "years_experience", "strengths",
                                         "weaknesses", "skills", "goals",
                                         "constraints", "job_description",
                                         "resume_text"}})
        profile.story_bank = list(data.get("story_bank", []))
        profile.style = {**profile.style, **data.get("style", {})}
        profile.topic_affinity = dict(data.get("topic_affinity", {}))
        profile.evidence_weights = {**profile.evidence_weights,
                                    **data.get("evidence_weights", {})}
        profile.interactions = int(data.get("interactions", 0))
        return profile

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: str) -> "IntervieweeProfile":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def _move(current: float, target: float, step: float) -> float:
    return round(max(0.0, min(1.0, current + step * (target - current))), 4)


@dataclass
class StyleAdapter:
    """Translate a profile into concrete generation parameters."""

    profile: IntervieweeProfile = field(default_factory=IntervieweeProfile)

    def parameters(self) -> Dict[str, Any]:
        style = self.profile.style
        return {
            "max_words": int(30 + style.get("verbosity", 0.55) * 110),
            "min_words": max(10, int(20 + style.get("verbosity", 0.55) * 25)),
            "formality": style.get("formality", 0.65),
            "hedging": style.get("hedging", 0.3),
            "concreteness": style.get("concreteness", 0.7),
            "enthusiasm": style.get("enthusiasm", 0.6),
            "technical_depth": style.get("technical_depth", 0.6),
            "people_focus": style.get("people_focus", 0.45),
            "evidence_weights": dict(self.profile.evidence_weights),
            "register": self.register(),
        }

    def register(self) -> str:
        formality = self.profile.style.get("formality", 0.65)
        enthusiasm = self.profile.style.get("enthusiasm", 0.6)
        if formality > 0.75:
            return "formal"
        if enthusiasm > 0.75 and formality < 0.6:
            return "conversational"
        if formality < 0.35:
            return "casual"
        return "professional"

    def sentence_style(self) -> Dict[str, Any]:
        return {
            "contractions": self.profile.style.get("formality", 0.65) < 0.55,
            "average_sentence_words": int(12 + self.profile.style.get("verbosity", 0.55) * 12),
            "lead_with_headline": self.profile.style.get("verbosity", 0.55) < 0.45,
        }
