"""The retrieval-augmented response composer.

This is where a question becomes an answer.  The composer:

1. picks a **headline** sentence that states the answer up front (interviewers
   reward this, and it makes the answer robust to being cut short),
2. selects **supporting content** -- knowledge-graph evidence, common-sense
   implications, the candidate's own story bank, and recalled memory,
3. adds a **trade-off or qualification** so the answer does not read as naive,
4. closes with an **empathetic, register-matched** line, and
5. records the metrics the evaluation layer needs.

Content comes from three sources, in priority order:

* the interviewee profile (their real strengths, skills and stories),
* the knowledge graph (definitions, relationships, alternatives),
* common-sense implications (what a competent answer must mention).

An LLM backend (:mod:`interviewgenie.generation.llm`) can replace the composer
entirely when an API key is configured; the composer's output is then used as
the grounded prompt and as the fallback if the model is unavailable.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..context.emotion import EmotionalIntelligence
from ..knowledge.commonsense import CommonsenseEngine, PlausibilityReport
from ..knowledge.retrieval import RetrievalResult
from ..logging import get_logger
from ..nlp.embeddings import sparse_cosine
from ..nlp.intent import IntentClassifier
from ..personalization.profile import IntervieweeProfile, StyleAdapter
from ..types import Analysis, Evidence, Response, Scorecard
from .templates import (PlanStep, ResponsePlanner, SurfaceRealiser, OPENERS,
                        CLOSERS, HEDGES, TRANSITIONS, _tidy)

LOG = get_logger("generation.composer")


# --------------------------------------------------------------------------- #
# Answer skeletons
# --------------------------------------------------------------------------- #
@dataclass
class AnswerSkeleton:
    """Intent-specific sentence patterns with named slots."""

    intent: str
    headline: Tuple[str, ...]
    support: Tuple[str, ...]
    tradeoff: Tuple[str, ...]


SKELETONS: Dict[str, AnswerSkeleton] = {
    "knowledge": AnswerSkeleton(
        "knowledge",
        headline=(
            "The short answer: {subject_a} and {subject_b} differ mainly in {contrast}.",
            "At a high level, {subject_a} is {definition_a}, whereas {subject_b} is {definition_b}.",
            "It comes down to {contrast}: {subject_a} optimises for {strength_a}, while {subject_b} optimises for {strength_b}.",
        ),
        support=(
            "In practice that means {implication}.",
            "You would reach for {subject_a} when {use_case_a}, and {subject_b} when {use_case_b}.",
            "{knowledge_sentence}",
        ),
        tradeoff=(
            "The trade-off is {contrast} against {opposite_contrast}.",
            "Where it bites: {tradeoff_note}.",
        ),
    ),
    "design": AnswerSkeleton(
        "design",
        headline=(
            "I would start from the requirements and work outwards.",
            "My approach is to pin down the contract first, then pick the simplest thing that holds.",
            "I would design it in three layers: the contract, the hot path, and the failure path.",
        ),
        support=(
            "For the hot path I would use {subject_a}, because {definition_a}.",
            "The pieces around it are {neighbours}, and I would keep each behind a clear interface.",
            "Capacity-wise I would sketch the numbers first: {capacity_note}.",
        ),
        tradeoff=(
            "The main trade-off is {contrast} versus {opposite_contrast}.",
            "The risk I would call out explicitly: {tradeoff_note}.",
        ),
    ),
    "self_introduction": AnswerSkeleton(
        "self_introduction",
        headline=(
            "I am {role_phrase}.",
            "Short version: I am {role_phrase}.",
            "I spend my time on {focus}.",
        ),
        support=(
            "Right now I am {current_focus}.",
            "Before that I {previous_focus}.",
            "What I care most about is {focus}.",
        ),
        tradeoff=(
            "If you only remember one thing about me, make it {focus}.",
        ),
    ),
    "background": AnswerSkeleton(
        "background",
        headline=(
            "I will take it in order.",
            "Three moves, each towards more ownership.",
        ),
        support=(
            "I started on {previous_focus}.",
            "Then I moved to {current_focus}.",
            "What I care most about is {focus}.",
        ),
        tradeoff=(
            "The through-line is {focus}.",
        ),
    ),
    "past_experience": AnswerSkeleton(
        "past_experience",
        headline=(
            "The example I would reach for is {story_title}.",
            "There is one project that maps directly onto this: {story_title}.",
            "I will use {story_title} as the example.",
        ),
        support=(
            "The situation was {story_situation}.",
            "What I actually did: {story_action}.",
            "The result: {story_result}.",
        ),
        tradeoff=(
            "The part I would do differently: {story_learning}.",
            "What it taught me: {story_learning}.",
        ),
    ),
    "self_assessment": AnswerSkeleton(
        "self_assessment",
        headline=(
            "Honest answer: {trait} is the one I would lead with.",
            "I would point at {trait}.",
            "The one I am actively working on is {trait}.",
        ),
        support=(
            "The evidence for that is {evidence}.",
            "{story_sentence}",
        ),
        tradeoff=(
            "The cost of that is {tradeoff_note}.",
            "What I am doing about it: {tradeoff_note}.",
        ),
    ),
    "people": AnswerSkeleton(
        "people",
        headline=(
            "The example I would reach for is {story_title}.",
            "I will use {story_title}, because it is the closest match.",
            "There is one situation that maps directly onto this: {story_title}.",
        ),
        support=(
            "The situation was {story_situation}.",
            "What I actually did: {story_action}.",
            "The result: {story_result}.",
        ),
        tradeoff=(
            "What I took from it: {story_learning}.",
            "The part I would do differently: {story_learning}.",
        ),
    ),
    "motivation": AnswerSkeleton(
        "motivation",
        headline=(
            "The honest answer is {reason}.",
            "Two things draw me to it: {reason}.",
            "It comes down to fit: {reason}.",
        ),
        support=(
            "{company_note}",
            "The work the team does on {neighbours} is the sort of thing I already spend my own time on.",
        ),
        tradeoff=(
            "What I would want to prove in the first ninety days: {tradeoff_note}.",
        ),
    ),
    "troubleshooting": AnswerSkeleton(
        "troubleshooting",
        headline=(
            "I would start by narrowing the blast radius before changing anything.",
            "My first move is to reproduce it and get a signal, not a guess.",
            "I would treat it as a measurement problem before a coding problem.",
        ),
        support=(
            "Then I would work outwards from {subject_a}, because {definition_a}.",
            "The usual suspects here are {neighbours}.",
            "I would instrument {capacity_note} so the next occurrence is cheaper.",
        ),
        tradeoff=(
            "The trap is {tradeoff_note}.",
            "I would rather roll back than debug live in production.",
        ),
    ),
    "coding": AnswerSkeleton(
        "coding",
        headline=(
            "I would start with the simplest correct version and then optimise.",
            "Let me talk through the approach before the syntax.",
        ),
        support=(
            "The structure would be {subject_a}: {definition_a}.",
            "The edge cases I would handle explicitly are {neighbours}.",
        ),
        tradeoff=("The complexity I would call out: {tradeoff_note}.",),
    ),
    "meta": AnswerSkeleton(
        "meta",
        headline=(
            "{meta_answer}",
        ),
        support=(),
        tradeoff=(),
    ),
    "logistics": AnswerSkeleton(
        "logistics",
        headline=(
            "{logistics_answer}",
        ),
        support=(),
        tradeoff=(),
    ),
    "default": AnswerSkeleton(
        "default",
        headline=(
            "Let me take that in turn.",
            "Here is how I would think about it.",
        ),
        support=(
            "The relevant pieces are {subject_a} and {neighbours}.",
            "{implication}",
        ),
        tradeoff=("The trade-off worth naming: {tradeoff_note}.",),
    ),
}


# --------------------------------------------------------------------------- #
# Composer
# --------------------------------------------------------------------------- #
@dataclass
class ResponseComposer:
    """Retrieval-augmented, style- and empathy-aware response generator."""

    profile: IntervieweeProfile = field(default_factory=IntervieweeProfile)
    style: StyleAdapter = field(default_factory=StyleAdapter)
    emotional: EmotionalIntelligence = field(default_factory=EmotionalIntelligence)
    commonsense: CommonsenseEngine = field(default_factory=CommonsenseEngine)
    planner: ResponsePlanner = field(default_factory=ResponsePlanner)
    realiser: SurfaceRealiser = field(default_factory=SurfaceRealiser)
    max_words: int = 90
    min_words: int = 12
    use_evidence: bool = True

    def __post_init__(self) -> None:
        self.style.profile = self.profile

    # -- main entry point ------------------------------------------------- #
    def compose(self, analysis: Analysis, retrieval: RetrievalResult,
                strategy: str = "direct", *, question_text: Optional[str] = None,
                memory: Optional[Sequence[str]] = None,
                feedback: Optional[str] = None) -> Response:
        started = time.perf_counter()
        question_text = question_text or analysis.text
        if feedback:
            self.profile.apply_feedback(feedback)

        intent = analysis.intent.name if analysis.intent else "default"
        params = self.style.parameters()

        guidance = self.emotional.respond_with(question_text)
        register = guidance.get("register", params.get("register", "professional"))

        headline = self._headline(intent, analysis, retrieval, strategy, question_text)
        support = self._support(intent, analysis, retrieval)
        tradeoff = self._tradeoff(intent, analysis, retrieval)

        opener = self._opener(register, strategy)
        closer = self._closer(register, strategy, guidance)

        sentences: List[str] = []
        plan_labels: List[str] = []
        if opener:
            sentences.append(opener)
            plan_labels.append("opener")
        if headline:
            sentences.append(headline)
            plan_labels.append("headline")
        sentences.extend(support)
        plan_labels.extend(["support"] * len(support))
        if tradeoff:
            sentences.append(tradeoff)
            plan_labels.append("tradeoff")
        if closer:
            sentences.append(closer)
            plan_labels.append("closer")

        sentences = _dedupe_sentences(sentences)
        text = _tidy(" ".join(s for s in sentences if s))
        max_words = params.get("max_words", self.max_words)
        if len(text.split()) > max_words:
            text = _truncate(text, max_words)

        evidence = list(retrieval.evidence[:6])
        story = getattr(self, "_current_story", None)
        if story and story.get("title"):
            evidence.append(Evidence(
                kind="profile",
                text=" ".join(str(story.get(k, "")) for k in
                              ("title", "situation", "action", "result")).strip(),
                source="interviewee profile",
                score=0.9,
            ))
        if self.profile.strengths or self.profile.skills:
            evidence.append(Evidence(
                kind="profile",
                text=" ".join(self.profile.strengths + self.profile.skills),
                source="interviewee profile",
                score=0.7,
            ))
        response = Response(
            text=text, turn_id=None, plan=plan_labels, evidence=evidence,
            tone=guidance.get("dominant_emotion", "neutral"),
            register=register,
            confidence=0.0,
        )
        response.validation = self._validate(analysis, response, retrieval)
        response.metrics = self._metrics(analysis, response, retrieval, question_text)
        response.confidence = self._confidence(response)
        response.latency_ms = (time.perf_counter() - started) * 1000.0
        self.profile.observe_answer(text, reward=response.confidence)
        return response

    # -- content ---------------------------------------------------------- #
    def _entities(self, retrieval: RetrievalResult, limit: int = 4) -> List[Any]:
        """Ranked, de-duplicated entities most relevant to the question.

        Entities whose name literally appears in the question come first; the
        rest are ordered by label priority (skills and concepts make better
        subjects than metrics or competencies).
        """
        question = (retrieval.question_text or "").lower()
        label_priority = {"SKILL": 0, "CONCEPT": 1, "COMPANY": 2, "ROLE": 3,
                          "COMPETENCY": 4, "METRIC": 5}
        seen: set[str] = set()
        ranked: List[Tuple[int, int, Any]] = []
        for index, entity in enumerate(retrieval.entities):
            name = str(entity.get("name", entity.id))
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            in_question = 0 if name.lower() in question else 1
            label_rank = label_priority.get(str(entity.label()), 6)
            ranked.append((in_question, label_rank + index * 0.01, entity))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [entity for _a, _b, entity in ranked][:limit]

    def _headline(self, intent: str, analysis: Analysis, retrieval: RetrievalResult,
                  strategy: str, question_text: str) -> str:
        if strategy == "clarify":
            return self._clarification(analysis)
        skeleton = SKELETONS.get(intent, SKELETONS["default"])
        entities = self._entities(retrieval)
        slots = self._slots(intent, analysis, retrieval, entities)

        if strategy == "question_back":
            return self._question_back(intent, analysis)

        for pattern in skeleton.headline:
            filled = _fill(pattern, slots)
            if filled and len(filled.split()) >= 6:
                return _tidy(filled)

        # fall back to a plain but useful headline
        if entities:
            names = ", ".join(str(e.get("name", e.id)) for e in entities[:3])
            return _tidy(f"The core of it is {names}: "
                         f"{self._first_definition(entities)}")
        return _tidy("Let me answer that directly and then show the reasoning.")

    def _support(self, intent: str, analysis: Analysis,
                 retrieval: RetrievalResult) -> List[str]:
        if not self.use_evidence:
            return []
        skeleton = SKELETONS.get(intent, SKELETONS["default"])
        entities = self._entities(retrieval)
        slots = self._slots(intent, analysis, retrieval, entities)
        out: List[str] = []
        for pattern in skeleton.support:
            filled = _fill(pattern, slots)
            if not filled:
                continue
            out.append(_tidy(filled))
            if len(out) >= 2:
                break
        if not out:
            for evidence in retrieval.evidence[:2]:
                if evidence.kind == "fact":
                    out.append(_tidy(f"To ground that: {evidence.text}"))
                    break
        return out

    def _tradeoff(self, intent: str, analysis: Analysis,
                  retrieval: RetrievalResult) -> str:
        skeleton = SKELETONS.get(intent, SKELETONS["default"])
        entities = self._entities(retrieval)
        slots = self._slots(intent, analysis, retrieval, entities)
        for pattern in skeleton.tradeoff:
            filled = _fill(pattern, slots)
            if filled and len(filled.split()) >= 5:
                return _tidy(filled)
        return ""

    # -- slot filling ----------------------------------------------------- #
    def _slots(self, intent: str, analysis: Analysis, retrieval: RetrievalResult,
               entities: Sequence[Any]) -> Dict[str, str]:
        names = _unique_names(entities)
        definitions = [_sentence(str(e.get("description", ""))) for e in entities]
        slots: Dict[str, str] = {
            "subject_a": names[0] if names else "",
            "subject_b": names[1] if len(names) > 1 else "",
            "definition_a": definitions[0] if definitions else "",
            "definition_b": definitions[1] if len(definitions) > 1 else "",
            "neighbour": names[2] if len(names) > 2 else (names[0] if names else ""),
            "neighbours": ", ".join(n for n in names[1:4]
                                    if n.lower() not in {names[0].lower()} ) if len(names) > 1 else "",
            "contrast": self._contrast(entities, analysis),
            "opposite_contrast": self._opposite_contrast(analysis),
            "strength_a": self._strength_of(entities, 0),
            "strength_b": self._strength_of(entities, 1),
            "use_case_a": self._use_case(entities, 0),
            "use_case_b": self._use_case(entities, 1),
            "implication": self._implication(analysis, retrieval),
            "capacity_note": self._capacity_note(analysis),
            "tradeoff_note": self._tradeoff_note(analysis, retrieval),
            "knowledge_sentence": self._knowledge_sentence(retrieval),
            "company_note": self._company_note(analysis, retrieval),
            "story_title": self._story_title(analysis),
            "story_situation": self._story_field("situation"),
            "story_action": self._story_field("action"),
            "story_result": self._story_field("result"),
            "meta_answer": self._meta_answer(analysis),
            "logistics_answer": self._logistics_answer(analysis),
            "story_learning": (self._story_field("learning")
                               or self._story_field("result")
                               or self._tradeoff_note(analysis, retrieval)),
            "story_sentence": self._story_sentence(analysis),
            "trait": self._trait(analysis),
            "role_phrase": self._role_phrase(),
            "focus": self._focus(),
            "current_focus": self._current_focus(),
            "previous_focus": self._previous_focus(),
            "strength": self.profile.strengths[0] if self.profile.strengths else "systems thinking",
            "weakness": self.profile.weaknesses[0] if self.profile.weaknesses else "depth in one narrow area",
            "reason": self._motivation_reason(analysis, retrieval),
            "evidence": self._profile_evidence(analysis),
        }
        return slots

    # -- individual slot builders ----------------------------------------- #
    def _trait(self, analysis: Analysis) -> str:
        """Pick a strength or a weakness depending on how the question is framed."""
        low = analysis.text.lower()
        wants_weakness = any(k in low for k in (
            "weakness", "improve", "struggle", "not good at", "mistake",
            "area you", "hardest", "regret", "do differently"))
        if wants_weakness:
            if self.profile.weaknesses:
                return self.profile.weaknesses[0]
            return "going deep in one narrow area before broadening out"
        if self.profile.strengths:
            return self.profile.strengths[0]
        return "turning ambiguous problems into a short list of concrete steps"

    def _story_sentence(self, analysis: Analysis) -> str:
        """A one-line concrete illustration for self-assessment answers."""
        story = getattr(self, "_current_story", None)
        if story and story.get("title"):
            action = str(story.get("action", "")).strip()
            if action:
                return f"A recent example: {action}"
        if self.profile.skills:
            return (f"You can see it in the work I have done with "
                    f"{', '.join(self.profile.skills[:3])}")
        return ""

    # -- profile-driven narrative ------------------------------------------ #
    def _role_phrase(self) -> str:
        years = (f"{self.profile.years_experience} years in"
                 if self.profile.years_experience else "working in")
        if self.profile.target_role and self.profile.skills:
            return (f"{self.profile.target_role} with about {years} "
                    f"{', '.join(self.profile.skills[:2])}")
        if self.profile.target_role:
            return f"{self.profile.target_role} with about {years} the space"
        if self.profile.skills:
            return f"an engineer working in {', '.join(self.profile.skills[:3])}"
        return "an engineer who likes hard production problems"

    def _focus(self) -> str:
        if self.profile.strengths:
            return self.profile.strengths[0]
        if self.profile.goals:
            return f"getting better at {self.profile.goals[0]}"
        return "making production systems boring"

    def _current_focus(self) -> str:
        if self.profile.strengths:
            return (f"focused on {self.profile.strengths[0]}")
        if self.profile.skills:
            return f"working mostly in {', '.join(self.profile.skills[:2])}"
        return "working on delivery reliability"

    def _previous_focus(self) -> str:
        story = getattr(self, "_current_story", None)
        if story and story.get("title"):
            return f"owned {story['title']}"
        if self.profile.skills:
            return f"was deep in {self.profile.skills[0]}"
        return "was on the platform side"

    # -- conversational repair and logistics -------------------------------- #
    def _meta_answer(self, analysis: Analysis) -> str:
        low = analysis.text.lower()
        if any(cue in low for cue in ("repeat", "say that again", "come again",
                                      "did not catch", "didn't catch", "pardon",
                                      "not sure i understood", "clarify",
                                      "what do you mean", "rephrase")):
            return ("Of course -- which part would you like me to go over again, the "
                    "approach or the specific project I mentioned?")
        if "questions for us" in low or "any questions" in low:
            return ("Yes, a couple. What does the team measure success on this quarter, "
                    "and what is the hardest problem this role is expected to own?")
        if "how are you" in low or "how's it going" in low or "how is it going" in low:
            return ("I am good, thank you -- a little nervous, but glad to be here and "
                    "looking forward to the conversation.")
        if "ready" in low or "shall we" in low or "can we start" in low:
            return "I am ready whenever you are."
        if "understand" in low or "make sense" in low or "clear" in low:
            return ("That is clear, thank you. Let me answer it directly and then show "
                    "the reasoning.")
        return "Happy to take that -- go ahead."

    def _logistics_answer(self, analysis: Analysis) -> str:
        low = analysis.text.lower()
        if any(cue in low for cue in ("salary", "compensation", "pay", "equity")):
            return ("I am looking for something in the senior band for this market, and I "
                    "am flexible depending on the equity mix and the scope of the role.")
        if any(cue in low for cue in ("start", "notice", "available", "when can you")):
            return ("I can give four weeks' notice, so realistically about a month from an "
                    "offer, and I am happy to be flexible if that helps.")
        if any(cue in low for cue in ("relocat", "remote", "hybrid", "onsite", "office",
                                      "commute")):
            return ("I am based here and happy to work in the office a few days a week; "
                    "remote is a bonus rather than a requirement for me.")
        if any(cue in low for cue in ("visa", "sponsor", "work authoris", "work authoriz")):
            return ("I am authorised to work here, so no sponsorship is needed.")
        if any(cue in low for cue in ("other offers", "interviewing", "competing",
                                      "other companies")):
            return ("I am in a couple of processes, but this is the one I am most "
                    "interested in, so I would rather move at your pace.")
        if any(cue in low for cue in ("contract", "freelance", "part time", "hours")):
            return "I am looking for a full-time role."
        return ("Happy to sort the details out -- what would be most useful for me to "
                "confirm?")

    def _knowledge_sentence(self, retrieval: RetrievalResult) -> str:
        """Describe the strongest retrieved path in plain prose."""
        for evidence in retrieval.evidence:
            if evidence.kind != "kg_path" or "->" not in evidence.text:
                continue
            parts = [p.strip() for p in evidence.text.split("->")]
            if len(parts) < 2:
                continue
            head = parts[0]
            if len(parts) == 2:
                return f"{head} is the piece that ties them together"
            middle = ", ".join(parts[1:-1])
            tail = parts[-1]
            return f"{head} sits alongside {middle} and connects through to {tail}"
        for evidence in retrieval.evidence:
            if evidence.kind == "fact":
                return _sentence(evidence.text)
        return ""

    def _company_note(self, analysis: Analysis, retrieval: RetrievalResult) -> str:
        companies = [e for e in retrieval.entities if e.label() == "COMPANY"]
        skills = [e for e in retrieval.entities if e.label() == "SKILL"]
        parts: List[str] = []
        if companies:
            company = companies[0]
            description = _sentence(str(company.get("description", "")))
            parts.append(f"{company.get('name')} is {description}" if description
                         else f"{company.get('name')} is a team I already follow closely")
            stack = [str(s.get("name")) for s in skills[:3]]
            if stack:
                parts.append(f"and the stack they run on -- {', '.join(stack)} -- is close to "
                             f"what I already work in day to day")
        elif skills:
            names = _unique_names(skills[:3])
            parts.append(f"the work sits on {', '.join(names)}, which is the layer I enjoy most")
        if not parts:
            return "the problems they describe are the ones I already work on"
        return ". ".join(p.rstrip(".") for p in parts) + "."

    def _capacity_note(self, analysis: Analysis) -> str:
        """Turn any numbers in the question into a capacity framing."""
        text = analysis.text.lower()
        for token, note in (
            ("billion", "billions of records means the storage and fan-out budget dominates"),
            ("million", "millions of users means the read path dominates and the write path must be cheap"),
            ("thousand", "a few thousand users means I would keep it boring and single-region"),
            ("qps", "the QPS figure in the question sets the hot-path budget"),
            ("rps", "the RPS figure in the question sets the hot-path budget"),
            ("real time", "real-time requirements mean the budget is latency, not throughput"),
            ("real-time", "real-time requirements mean the budget is latency, not throughput"),
        ):
            if token in text:
                return note
        return "the read/write ratio drives the storage and fan-out budget"

    def _strength_of(self, entities: Sequence[Any], index: int) -> str:
        if index >= len(entities):
            return "correctness"
        category = str(entities[index].get("category", ""))
        mapping = {
            "language": "developer velocity",
            "framework": "developer velocity",
            "database": "query flexibility",
            "infrastructure": "operational control",
            "cloud": "operational leverage",
            "data": "analytical depth",
            "ml": "model quality",
            "practice": "consistency",
            "concept": "clarity",
        }
        return mapping.get(category, "correctness")

    def _use_case(self, entities: Sequence[Any], index: int) -> str:
        if index >= len(entities):
            return "the requirements point that way"
        node = entities[index]
        name = str(node.get("name", node.id))
        category = str(node.get("category", ""))
        mapping = {
            "database": f"you need strong query semantics from {name}",
            "infrastructure": f"you are operating {name} at scale",
            "cloud": f"{name} gives you leverage you do not want to build yourself",
            "language": f"the team already ships {name}",
            "ml": f"the problem is fundamentally a learning problem",
            "practice": f"the team needs the consistency {name} buys",
            "concept": f"you need what {name} gives you -- {_sentence(str(node.get('description', ''))) or 'a clear mental model'}",
        }
        return mapping.get(category, f"the constraints point at {name}")

    def _contrast(self, entities: Sequence[Any], analysis: Analysis) -> str:
        """Name the real axis on which the mentioned concepts differ."""
        names = _unique_names(entities)
        text = analysis.text.lower()
        if len(names) >= 2:
            pair = {names[0].lower(), names[1].lower()}
            known = {
                frozenset({"tcp", "udp"}): "reliability versus overhead",
                frozenset({"rest", "graphql"}): "predictability versus query flexibility",
                frozenset({"microservices", "monolith"}): "independent deployability versus operational simplicity",
                frozenset({"sql", "nosql"}): "schema discipline versus flexible documents",
                frozenset({"postgresql", "mysql"}): "feature breadth versus operational familiarity",
                frozenset({"pytorch", "tensorflow"}): "research ergonomics versus production serving",
                frozenset({"eventual consistency", "strong consistency"}): "availability versus read-your-writes",
                frozenset({"concurrency", "parallelism"}): "structure versus simultaneity",
                frozenset({"mutex", "semaphore"}): "exclusive ownership versus bounded capacity",
                frozenset({"kafka", "rabbitmq"}): "replayable log versus low-latency routing",
                frozenset({"redis", "memcached"}): "data structures versus raw speed",
                frozenset({"docker", "podman"}): "daemon convenience versus daemonless isolation",
                frozenset({"aws", "gcp"}): "breadth of services versus data and ML depth",
            }
            for key, phrase in known.items():
                if key <= pair:
                    return phrase
        if "design" in text or "architect" in text or "scale" in text:
            return "availability against consistency"
        if "difference" in text or "versus" in text or " vs " in text:
            return "the guarantees each one gives you"
        return "the guarantees each option gives you"

    def _opposite_contrast(self, analysis: Analysis) -> str:
        text = analysis.text.lower()
        if "design" in text or "architect" in text:
            return "operational simplicity"
        if "difference" in text:
            return "raw speed"
        return "simplicity"

    def _implication(self, analysis: Analysis, retrieval: RetrievalResult) -> str:
        """A concrete consequence, phrased as a statement about the design."""
        intent = analysis.intent.name if analysis.intent else ""
        if intent in {"self_assessment", "past_experience", "people", "motivation", "goals"}:
            return self._profile_evidence(analysis)
        implications = self.commonsense.implications(analysis.text)
        # drop meta-implications about the candidate -- they are coaching notes
        concrete = [i for i in implications if "candidate" not in i.lower()]
        if concrete:
            note = concrete[0]
            return _lower(note)
        entities = self._entities(retrieval)
        if entities:
            node = entities[0]
            description = _sentence(str(node.get("description", "")))
            if description:
                return f"{node.get('name')} has to handle {description}"
        return "the design has to survive its own failure modes"

    def _tradeoff_note(self, analysis: Analysis, retrieval: RetrievalResult) -> str:
        entities = self._entities(retrieval)
        names = _unique_names(entities)
        text = analysis.text.lower()
        if "design" in text or "architect" in text or "scale" in text:
            if names:
                return (f"every extra hop through {names[0]} buys correctness at the cost "
                        f"of latency, so I would keep the number of network calls on the "
                        f"hot path to one")
            return "every extra hop buys correctness at the cost of latency"
        if "difference" in text or "versus" in text:
            if len(names) >= 2:
                return (f"{names[0]} is the safer default, but {names[1]} wins as soon as "
                        f"that constraint stops binding")
            return "the safer default wins until the constraint stops binding"
        if names:
            return (f"the risk is over-engineering {names[0]} before the numbers justify it")
        return "the risk is over-engineering before the numbers justify it"

    def _first_definition(self, entities: Sequence[Any]) -> str:
        for node in entities:
            description = str(node.get("description", "")).strip()
            if description:
                return _sentence(description)
        return "the details matter more than the label"

    # -- profile-driven content ------------------------------------------- #
    def _profile_evidence(self, analysis: Analysis) -> str:
        if self.profile.skills:
            return f"the work I have done with {', '.join(self.profile.skills[:3])}"
        if self.profile.strengths:
            return f"the track record in {self.profile.strengths[0].lower()}"
        return "the projects in my background"

    def _story_title(self, analysis: Analysis) -> str:
        topic = analysis.intent.topic if analysis.intent else ""
        stories = self.profile.stories_for(topic or analysis.text)
        if not stories:
            stories = self.profile.story_bank[:1]
        if stories:
            self._current_story = stories[0]
            return stories[0].get("title", "a recent project")
        self._current_story = None
        return ""

    def _story_field(self, field_name: str) -> str:
        story = getattr(self, "_current_story", None)
        if not story:
            return ""
        return str(story.get(field_name, "")).strip()

    def _motivation_reason(self, analysis: Analysis, retrieval: RetrievalResult) -> str:
        reasons: List[str] = []
        if self.profile.strengths:
            reasons.append(f"I get to keep doing what I am best at -- "
                           f"{', '.join(self.profile.strengths[:2])}")
        if self.profile.goals:
            reasons.append(f"it moves me towards {self.profile.goals[0]}")
        if self.profile.target_role:
            reasons.append(f"it is the {self.profile.target_role} scope I have been building towards")
        if len(reasons) >= 2:
            return "; ".join(reasons[:2])
        entities = self._entities(retrieval)
        for node in entities[:2]:
            name = str(node.get("name", node.id))
            description = str(node.get("description", ""))
            if description:
                reasons.append(f"the chance to work on {name}, where {_sentence(description)}")
        if self.profile.target_role:
            reasons.append(f"it is the {self.profile.target_role} scope I have been building towards")
        if self.profile.goals:
            reasons.append(f"it moves me towards {self.profile.goals[0]}")
        if not reasons:
            reasons.append("the problem is one I already spend my own time on")
        return "; ".join(reasons[:2])

    # -- conversation repair ---------------------------------------------- #
    def _clarification(self, analysis: Analysis) -> str:
        return ("Just so I answer the right thing -- are you asking about the "
                "approach, or about a specific project I mentioned?")

    def _question_back(self, intent: str, analysis: Analysis) -> str:
        questions = {
            "motivation": "What does success in the first year look like for your team?",
            "design": "Which constraint matters most here -- latency, cost, or time to ship?",
            "culture": "How does the team prefer to make decisions when there is disagreement?",
            "default": "Would it help if I went deeper on any part of that?",
        }
        return questions.get(intent, questions["default"])

    # -- register --------------------------------------------------------- #
    def _opener(self, register: str, strategy: str) -> str:
        if strategy in {"clarify"}:
            return ""
        options = OPENERS.get(register, OPENERS["professional"])
        return options[0] if options else ""

    def _closer(self, register: str, strategy: str, guidance: Dict[str, str]) -> str:
        if strategy == "question_back":
            return self._question_back("default", Analysis(text=""))
        options = CLOSERS.get(register, CLOSERS["professional"])
        return options[0] if options else ""

    # -- validation & metrics --------------------------------------------- #
    def _validate(self, analysis: Analysis, response: Response,
                  retrieval: RetrievalResult) -> Dict[str, Any]:
        categories = [str(e.get("category", "")) for e in retrieval.entities]
        report = self.commonsense.check(response.text, analysis.text, categories)
        return {
            "plausibility": report.score,
            "violations": [v["rule"] for v in report.violations],
            "satisfied": report.satisfied,
            "words": len(response.text.split()),
            "has_question_back": "?" in response.text,
        }

    def _metrics(self, analysis: Analysis, response: Response,
                 retrieval: RetrievalResult, question_text: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics["groundedness"] = _groundedness(response, retrieval)
        metrics["personalization"] = self.profile.personalisation_score(response.text)
        metrics["engagement"] = self.emotional.engagement_score(response.text)
        metrics["empathy"] = self.emotional.empathy_score(response.text, question_text)
        metrics["relevance"] = _relevance(response.text, question_text, retrieval)
        metrics["accuracy"] = _accuracy(response.text, retrieval)
        metrics["fluency"] = _fluency(response.text)
        metrics["concision"] = _concision(response.text)
        return metrics

    def _confidence(self, response: Response) -> float:
        validation = response.validation
        metrics = response.metrics
        score = 0.25
        score += 0.30 * metrics.get("groundedness", 0.0)
        score += 0.20 * metrics.get("accuracy", 0.0)
        score += 0.15 * validation.get("plausibility", 0.0)
        score += 0.10 * metrics.get("fluency", 0.0)
        score -= 0.15 * min(1.0, len(validation.get("violations", [])) / 3.0)
        return round(max(0.0, min(1.0, score)), 4)


# --------------------------------------------------------------------------- #
# scoring helpers
# --------------------------------------------------------------------------- #
def _groundedness(response: Response, retrieval: RetrievalResult) -> float:
    """Fraction of the answer's sentences backed by retrieved evidence.

    Unused evidence must not count against the answer, so the score is computed
    sentence-by-sentence rather than as a ratio over the evidence list.
    """
    if not response.evidence:
        return 0.0
    sentences = [s for s in re.split(r"[.!?]+", response.text) if len(s.split()) >= 3]
    if not sentences:
        return 0.0
    evidence_bags = [
        {w.lower() for w in e.text.split() if len(w) > 4}
        for e in response.evidence
    ]
    evidence_bags = [bag for bag in evidence_bags if bag]
    if not evidence_bags:
        return 0.0
    supported = 0
    for sentence in sentences:
        words = {w.lower() for w in sentence.split() if len(w) > 4}
        if any(words & bag for bag in evidence_bags):
            supported += 1
    return round(supported / len(sentences), 4)


def _relevance(answer: str, question: str, retrieval: RetrievalResult) -> float:
    question_terms = {w.lower() for w in question.split() if len(w) > 4}
    answer_terms = {w.lower() for w in answer.split() if len(w) > 4}
    if not question_terms or not answer_terms:
        return 0.0
    overlap = len(question_terms & answer_terms) / len(question_terms)
    entity_names = {str(e.get("name", e.id)).lower() for e in retrieval.entities}
    entity_hits = sum(1 for name in entity_names if name in answer.lower())
    return round(max(0.0, min(1.0, 0.6 * overlap + 0.1 * min(3, entity_hits))), 4)


def _accuracy(answer: str, retrieval: RetrievalResult) -> float:
    """How well the answer's concepts match the retrieved knowledge.

    Measured against the evidence that was actually used, so an answer grounded
    in the candidate's own story is not penalised for graph paths it never cited.
    """
    evidence = [e for e in (retrieval.evidence or [])]
    answer_terms = {w.lower() for w in answer.split() if len(w) > 4}
    if not evidence or not answer_terms:
        return 0.0
    hits = 0
    for item in evidence:
        terms = {w.lower() for w in item.text.split() if len(w) > 4}
        if terms & answer_terms:
            hits += 1
    denominator = min(len(evidence), 4)
    return round(min(1.0, hits / max(1, denominator)), 4)


def _fluency(text: str) -> float:
    """Cheap fluency proxy: sentence length variance and repetition penalty."""
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) < 2:
        return 0.9 if 4 <= lengths[0] <= 40 else 0.5
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    score = 1.0
    if variance < 4:
        score -= 0.15                     # monotonous
    if mean > 35:
        score -= 0.2                      # run-on
    if mean < 5:
        score -= 0.2                      # choppy
    words = text.lower().split()
    if words:
        repeats = sum(1 for w in set(words) if words.count(w) > 4)
        score -= 0.05 * repeats
    return round(max(0.0, min(1.0, score)), 4)


def _concision(text: str) -> float:
    words = len(text.split())
    if words <= 60:
        return 1.0
    if words >= 160:
        return 0.0
    return round(1.0 - (words - 60) / 100.0, 4)


def _sentence(text: str) -> str:
    """First sentence of ``text``, lower-cased so it can be embedded mid-sentence."""
    text = (text or "").strip()
    if not text:
        return ""
    text = text.split(". ")[0].strip().rstrip(".")
    if not text:
        return ""
    return text[0].lower() + text[1:] if text[0].isupper() else text


def _unique_names(entities: Sequence[Any]) -> List[str]:
    """De-duplicated node names, preserving order."""
    seen: set[str] = set()
    out: List[str] = []
    for entity in entities:
        name = str(entity.get("name", entity.id)).strip()
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            out.append(name)
    return out


_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")


def _lower(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return text[0].lower() + text[1:] if text[0].isupper() else text


def _fill(pattern: str, slots: Dict[str, str]) -> Optional[str]:
    """Format ``pattern`` only when every referenced slot has a usable value."""
    placeholders = _PLACEHOLDER_RE.findall(pattern)
    if any(not str(slots.get(name, "")).strip() for name in placeholders):
        return None
    try:
        filled = pattern.format(**slots)
    except (KeyError, IndexError, ValueError):
        return None
    if _has_unfilled(filled):
        return None
    return filled


def _dedupe_sentences(sentences: Sequence[str]) -> List[str]:
    """Drop sentences whose normalised form has already been emitted."""
    seen: set[str] = set()
    out: List[str] = []
    for sentence in sentences:
        key = re.sub(r"\W+", " ", sentence.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(sentence)
    return out


def _has_unfilled(text: str) -> bool:
    return bool(re.search(r"\{[a-z_]+\}|None|nan", text))


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    boundary = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if boundary > len(truncated) * 0.5:
        return truncated[: boundary + 1]
    return truncated.rstrip(",;: ") + "."
