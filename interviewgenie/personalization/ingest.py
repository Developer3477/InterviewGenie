"""Turn a pasted résumé and job description into a populated profile.

Every serious live-interview copilot personalises answers from the candidate's
résumé and the job description.  This module does that with no third-party
dependencies: it extracts the facts that actually change an answer -- skills,
seniority, years of experience, target role and company, education, and STAR
stories from résumé bullets -- and writes them into an
:class:`~interviewgenie.personalization.profile.IntervieweeProfile`.

Everything is heuristic and deliberately conservative: it only records what it
can point at in the text, because a hallucinated skill is worse than a missing
one when the answer is read aloud in an interview.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..logging import get_logger
from .profile import IntervieweeProfile

LOG = get_logger("personalization.ingest")

#: Technologies and practices recognised in résumés and job descriptions.  Kept
#: broad on purpose: the goal is to catch what the candidate typed, not to be a
#: complete taxonomy.
SKILL_GAZETTEER: Tuple[str, ...] = (
    # languages
    "python", "java", "javascript", "typescript", "golang", "go", "rust", "c++",
    "c#", "c", "ruby", "php", "swift", "kotlin", "scala", "elixir", "haskell",
    "r", "matlab", "perl", "dart", "objective-c", "shell", "bash", "powershell",
    # web / api
    "react", "angular", "vue", "svelte", "next.js", "node.js", "express",
    "django", "flask", "fastapi", "spring", "spring boot", "rails", "laravel",
    "graphql", "rest", "grpc", "websocket", "html", "css", "sass", "tailwind",
    # data
    "sql", "postgresql", "mysql", "sqlite", "mongodb", "redis", "cassandra",
    "dynamodb", "elasticsearch", "kafka", "rabbitmq", "sqs", "spark", "hadoop",
    "airflow", "dbt", "snowflake", "bigquery", "redshift", "pandas", "numpy",
    "scipy", "pytorch", "tensorflow", "scikit-learn", "keras", "xgboost",
    "machine learning", "deep learning", "nlp", "computer vision", "llm",
    "transformers", "rag", "fine-tuning",
    # infra / cloud
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform",
    "ansible", "pulumi", "helm", "jenkins", "circleci", "github actions",
    "gitlab ci", "argo", "istio", "prometheus", "grafana", "datadog",
    "opentelemetry", "nginx", "linux", "ci/cd", "devops", "sre", "microservices",
    "serverless", "lambda", "ec2", "s3", "rds", "cloudformation",
    # concepts / practice
    "system design", "distributed systems", "data structures", "algorithms",
    "object oriented", "functional programming", "test driven development",
    "unit testing", "integration testing", "code review", "agile", "scrum",
    "kanban", "observability", "monitoring", "caching", "load balancing",
    "sharding", "replication", "message queues", "event driven", "domain driven",
    "security", "oauth", "jwt", "encryption", "accessibility", "performance",
    "optimisation", "optimization", "mentoring", "leadership", "architecture",
)

#: Seniority markers, most junior first.
SENIORITY_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("intern", "intern"), ("trainee", "intern"), ("graduate", "junior"),
    ("junior", "junior"), ("entry level", "junior"), ("associate", "junior"),
    ("mid level", "mid"), ("mid-level", "mid"),
    ("senior", "senior"), ("sr.", "senior"), ("staff", "staff"),
    ("principal", "principal"), ("lead", "lead"), ("head of", "lead"),
    ("manager", "manager"), ("director", "director"), ("vp", "director"),
    ("chief", "director"),
)

#: Degree / education markers.
EDUCATION_MARKERS: Tuple[str, ...] = (
    "b.tech", "btech", "b.e.", "b.sc", "bsc", "b.a.", "bachelor", "bs ", "ba ",
    "m.tech", "mtech", "m.sc", "msc", "m.s.", "master", "mba", "phd", "ph.d",
    "doctorate", "diploma", "certification", "certificate",
)

#: Verbs that typically open an accomplishment bullet in a résumé.
ACCOMPLISHMENT_VERBS: Tuple[str, ...] = (
    "led", "built", "designed", "shipped", "launched", "reduced", "increased",
    "improved", "migrated", "scaled", "automated", "implemented", "created",
    "developed", "drove", "delivered", "owned", "architected", "refactored",
    "optimised", "optimized", "introduced", "established", "mentored",
    "coordinated", "negotiated", "resolved", "debugged", "instrumented",
)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")
PHONE_RE = re.compile(r"(?:\+\d{1,3}[- ]?)?(?:\(\d{2,4}\)|\d{2,4})[- ]?\d{3,4}[- ]?\d{3,4}")
URL_RE = re.compile(r"https?://[^\s,)]+|(?:www\.|github\.com/|linkedin\.com/in/)[^\s,)]+", re.I)
YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b", re.I)
BULLET_RE = re.compile(r"^\s*(?:[-*•·–]|\d+[.)])\s+(.*)$")
SECTION_RE = re.compile(r"^\s*([A-Za-z][A-Za-z /&-]{2,30})\s*:?\s*$")
REQUIREMENT_HINT_RE = re.compile(
    r"\b(?:require[ds]?|must have|should have|nice to have|preferred|"
    r"responsibilit|qualification|you will|we are looking|looking for)\b", re.I)

_SKILL_INDEX = {s.lower(): s for s in SKILL_GAZETTEER}
_SKILL_INDEX.update({s.lower().replace(".", ""): s for s in SKILL_GAZETTEER if "." in s})


@dataclass
class IngestionReport:
    """What was extracted, so the UI can show the user what was understood."""

    skills: List[str] = field(default_factory=list)
    seniority: str = ""
    years_experience: Optional[int] = None
    target_role: str = ""
    target_company: str = ""
    education: List[str] = field(default_factory=list)
    stories: int = 0
    jd_requirements: List[str] = field(default_factory=list)
    contacts: Dict[str, List[str]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skills": self.skills, "seniority": self.seniority,
            "years_experience": self.years_experience,
            "target_role": self.target_role,
            "target_company": self.target_company,
            "education": self.education, "stories": self.stories,
            "jd_requirements": self.jd_requirements,
            "contacts": self.contacts, "notes": self.notes,
        }


def _lines(text: str) -> List[str]:
    return [ln.rstrip() for ln in (text or "").splitlines()]


def _clean(phrase: str) -> str:
    return re.sub(r"\s+", " ", (phrase or "").strip()).strip(" ,;:-–—")


def find_skills(text: str) -> List[str]:
    """Every gazetteer skill that appears as a whole word/phrase in ``text``."""
    low = f" {text.lower()} "
    # normalise punctuation so "node.js," and "Node JS" both match
    normalised = re.sub(r"[^a-z0-9+.+#/ ]+", " ", low)
    normalised = re.sub(r"\s+", " ", normalised)
    found: List[str] = []
    for skill in SKILL_GAZETTEER:
        variants = {skill.lower(), skill.lower().replace(".", "")}
        for variant in variants:
            if re.search(rf"(?<![a-z0-9+#]){re.escape(variant)}(?![a-z0-9+#])",
                         normalised):
                found.append(_SKILL_INDEX.get(skill.lower(), skill))
                break
    # de-duplicate, longest first so "machine learning" beats "learning"
    return sorted(dict.fromkeys(found), key=lambda s: (-len(s), s))


def find_seniority(text: str) -> str:
    low = text.lower()
    best, best_rank = "", -1
    for marker, level in SENIORITY_MARKERS:
        rank = [s for s, _ in SENIORITY_MARKERS].index(marker)
        if re.search(rf"(?<![a-z]){re.escape(marker)}(?![a-z])", low):
            if rank > best_rank:
                best, best_rank = level, rank
    return best


def find_years(text: str) -> Optional[int]:
    years = [int(m) for m in YEARS_RE.findall(text)]
    return max(years) if years else None


def find_contacts(text: str) -> Dict[str, List[str]]:
    contacts: Dict[str, List[str]] = {}
    emails = EMAIL_RE.findall(text)
    if emails:
        contacts["emails"] = list(dict.fromkeys(emails))[:3]
    urls = [u.rstrip(".,);") for u in URL_RE.findall(text)]
    if urls:
        contacts["links"] = list(dict.fromkeys(urls))[:4]
    return contacts


def find_education(text: str) -> List[str]:
    found: List[str] = []
    for line in _lines(text):
        low = line.lower()
        if any(marker in low for marker in EDUCATION_MARKERS):
            cleaned = _clean(re.sub(r"^\s*(?:[-*•·–]|\d+[.)])\s*", "", line))
            if 3 < len(cleaned) < 160:
                found.append(cleaned)
    return list(dict.fromkeys(found))[:4]


def find_role_title(text: str, *, prefer_jd: bool = False) -> str:
    """Best guess at the role being interviewed for."""
    lines = _lines(text)
    # explicit "Role: X" / "Position: X" wins outright
    for line in lines[:40]:
        match = re.match(r"^\s*(?:role|position|title|job title)\s*[:\-]\s*(.+)$",
                         line, re.I)
        if match:
            candidate = _clean(match.group(1))
            if 2 < len(candidate) < 80:
                return candidate
    # a short title-like line near the top
    for line in lines[:12]:
        cleaned = _clean(re.sub(r"^\s*(?:[-*•·–]|\d+[.)])\s*", "", line))
        if not cleaned or len(cleaned) > 70 or cleaned.endswith("."):
            continue
        low = cleaned.lower()
        if any(word in low for word in (
                "engineer", "developer", "scientist", "designer", "manager",
                "analyst", "architect", "consultant", "lead", "intern",
                "specialist", "administrator", "researcher", "product")):
            return cleaned
    return ""


def find_company(text: str) -> str:
    """Look for 'at <Company>' or a 'Company: X' header."""
    for line in _lines(text)[:40]:
        match = re.match(r"^\s*(?:company|employer|organisation|organization|at)\s*[:\-]\s*(.+)$",
                         line, re.I)
        if match:
            candidate = _clean(match.group(1))
            if 1 < len(candidate) < 60:
                return candidate
    match = re.search(r"\bat\s+([A-Z][A-Za-z0-9&.\- ]{1,40})", text)
    if match:
        candidate = _clean(match.group(1))
        # avoid "at least", "at the", "at this point"
        if candidate.lower() not in {"least", "the", "this", "a", "an", "present",
                                     "scale", "once", "any", "no", "risk"}:
            return candidate
    return ""


def find_stories(text: str, limit: int = 8) -> List[Dict[str, str]]:
    """Derive STAR-ish stories from résumé accomplishment bullets.

    A bullet becomes a story when it opens with an accomplishment verb.  The
    sentence is kept verbatim as the *action* -- inventing situation/task around
    it would put words in the candidate's mouth.
    """
    stories: List[Dict[str, str]] = []
    for line in _lines(text):
        match = BULLET_RE.match(line)
        body = _clean(match.group(1) if match else line)
        if not body or len(body) < 25:
            continue
        first = body.split()[0].lower().strip(",.;:")
        if first in ACCOMPLISHMENT_VERBS or first.rstrip("ed") in ACCOMPLISHMENT_VERBS:
            title = " ".join(body.split()[:9])
            stories.append({
                "title": title if len(title) < 90 else title[:87] + "...",
                "situation": "",
                "task": "",
                "action": body,
                "result": "",
            })
        if len(stories) >= limit:
            break
    return stories


def find_requirements(text: str, limit: int = 12) -> List[str]:
    """Pull requirement-ish lines out of a job description."""
    found: List[str] = []
    for line in _lines(text):
        cleaned = _clean(re.sub(r"^\s*(?:[-*•·–]|\d+[.)])\s*", "", line))
        if not cleaned or len(cleaned) < 12:
            continue
        low = cleaned.lower()
        if REQUIREMENT_HINT_RE.search(low) or BULLET_RE.match(line):
            if cleaned not in found:
                found.append(cleaned)
        if len(found) >= limit:
            break
    return found


def ingest(resume_text: str = "", job_description: str = "",
           profile: Optional[IntervieweeProfile] = None
           ) -> Tuple[IntervieweeProfile, IngestionReport]:
    """Populate ``profile`` (or a fresh one) from résumé and JD text.

    Existing profile facts are merged, never overwritten with blanks, so a
    candidate can refine the result by hand afterwards.
    """
    profile = profile or IntervieweeProfile()
    report = IngestionReport()

    resume_text = resume_text or ""
    job_description = job_description or ""

    if resume_text:
        profile.resume_text = resume_text.strip()[:20000]
        report.skills = find_skills(resume_text)
        report.years_experience = find_years(resume_text)
        report.seniority = find_seniority(resume_text)
        report.education = find_education(resume_text)
        report.contacts = find_contacts(resume_text)
        stories = find_stories(resume_text)
        report.stories = len(stories)

        facts: Dict[str, Any] = {}
        if report.skills:
            facts["skills"] = report.skills[:24]
        if report.years_experience is not None:
            facts["years_experience"] = report.years_experience
        if report.seniority:
            facts["seniority"] = report.seniority
        if report.education:
            facts["constraints"] = report.education[:3]
        profile.update_facts(**facts)
        for story in stories:
            profile.add_story(**story)
        if not profile.name or profile.name == "Candidate":
            name = _guess_name(resume_text)
            if name:
                profile.name = name
                report.notes.append(f"name read as {name!r}")

    if job_description:
        profile.job_description = job_description.strip()[:20000]
        jd_skills = find_skills(job_description)
        report.jd_requirements = find_requirements(job_description)
        role = find_role_title(job_description, prefer_jd=True)
        if role:
            report.target_role = role
            profile.update_facts(target_role=role)
        company = find_company(job_description)
        if company:
            report.target_company = company
            profile.update_facts(target_company=company)
        seniority = find_seniority(job_description)
        if seniority and not profile.seniority:
            profile.update_facts(seniority=seniority)
        if jd_skills:
            # JD skills are what the interviewer will probe: keep them visible.
            merged = list(dict.fromkeys((profile.skills or []) + jd_skills[:16]))
            profile.update_facts(skills=merged[:32])
            if not report.skills:
                report.skills = jd_skills[:16]

    if not report.skills and not report.jd_requirements:
        report.notes.append(
            "nothing recognised -- paste the raw text of the résumé and the "
            "job posting, not a summary")
    return profile, report


def _guess_name(text: str) -> str:
    """Read a name off the first plausible line of a résumé."""
    for line in _lines(text)[:6]:
        cleaned = _clean(line)
        if not cleaned or "@" in cleaned or any(ch.isdigit() for ch in cleaned):
            continue
        words = cleaned.split()
        if 1 < len(words) <= 4 and all(w[:1].isupper() for w in words if w.isalpha()):
            low = cleaned.lower()
            if not any(marker in low for marker in ("resume", "cv", "curriculum")):
                return cleaned
    return ""


def summarise(profile: IntervieweeProfile) -> str:
    """A one-paragraph description of what the system knows, for the UI."""
    bits: List[str] = []
    if profile.name and profile.name != "Candidate":
        bits.append(profile.name)
    if profile.target_role:
        bits.append(f"targeting {profile.target_role}")
    if profile.target_company:
        bits.append(f"at {profile.target_company}")
    if profile.years_experience:
        bits.append(f"{profile.years_experience} years' experience")
    if profile.skills:
        shown = ", ".join(profile.skills[:6])
        more = f" (+{len(profile.skills) - 6} more)" if len(profile.skills) > 6 else ""
        bits.append(f"skills: {shown}{more}")
    if profile.story_bank:
        bits.append(f"{len(profile.story_bank)} STAR stories")
    return "; ".join(bits) if bits else "no profile information yet"
