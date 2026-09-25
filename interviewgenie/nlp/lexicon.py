"""Hand-built English lexicons: stop words, function words and POS priors.

Keeping these in code (rather than downloading corpora) is what lets
InterviewGenie run on a bare standard-library interpreter.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

# --------------------------------------------------------------------------- #
# Penn-Treebank-ish tag set used throughout the pipeline
# --------------------------------------------------------------------------- #
TAG_NOUN_SINGULAR = "NN"
TAG_NOUN_PLURAL = "NNS"
TAG_PROPER_SINGULAR = "NNP"
TAG_PROPER_PLURAL = "NNPS"
TAG_VERB_BASE = "VB"
TAG_VERB_PAST = "VBD"
TAG_VERB_GERUND = "VBG"
TAG_VERB_PAST_PART = "VBN"
TAG_VERB_3SG = "VBZ"
TAG_VERB_NON3SG = "VBP"
TAG_ADJ = "JJ"
TAG_ADJ_COMP = "JJR"
TAG_ADJ_SUP = "JJS"
TAG_ADV = "RB"
TAG_ADV_COMP = "RBR"
TAG_ADV_SUP = "RBS"
TAG_PREP = "IN"
TAG_DET = "DT"
TAG_PREDET = "PDT"
TAG_PRONOUN = "PRP"
TAG_POSS_PRONOUN = "PRP$"
TAG_WH_PRONOUN = "WP"
TAG_WH_POSS = "WP$"
TAG_WH_DET = "WDT"
TAG_WH_ADV = "WRB"
TAG_MODAL = "MD"
TAG_CONJ = "CC"
TAG_CARDINAL = "CD"
TAG_EXISTENTIAL = "EX"
TAG_TO = "TO"
TAG_PARTICLE = "RP"
TAG_INTERJ = "UH"
TAG_POSSESSIVE = "POS"
TAG_SYMBOL = "SYM"
TAG_FOREIGN = "FW"
TAG_LIST_ITEM = "LS"
TAG_PUNCT = "PUNCT"

#: tags that behave like nouns for downstream consumers
NOUN_TAGS = frozenset({TAG_NOUN_SINGULAR, TAG_NOUN_PLURAL, TAG_PROPER_SINGULAR, TAG_PROPER_PLURAL})
VERB_TAGS = frozenset({TAG_VERB_BASE, TAG_VERB_PAST, TAG_VERB_GERUND, TAG_VERB_PAST_PART,
                       TAG_VERB_3SG, TAG_VERB_NON3SG})
ADJ_TAGS = frozenset({TAG_ADJ, TAG_ADJ_COMP, TAG_ADJ_SUP})
ADV_TAGS = frozenset({TAG_ADV, TAG_ADV_COMP, TAG_ADV_SUP})

# --------------------------------------------------------------------------- #
# Stop words
# --------------------------------------------------------------------------- #
STOP_WORDS: FrozenSet[str] = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your
yours yourself yourselves also just like get got really much many may might
will shall must can need
""".split())

#: words that carry conversational weight and must *not* be dropped as noise
DISCOURSE_MARKERS = frozenset({
    "okay", "ok", "alright", "right", "sure", "well", "so", "now", "anyway",
    "actually", "basically", "honestly", "look", "listen", "great", "good",
    "interesting", "thanks", "thank", "please", "yes", "no", "yeah", "yep",
})

# --------------------------------------------------------------------------- #
# Closed-class lexicons (high confidence, no context needed)
# --------------------------------------------------------------------------- #
DETERMINERS: FrozenSet[str] = frozenset("""
the a an this that these those every each some any no another both neither
either enough much most more few fewer less least several all any both half
many such what whatever which whichever whose
""".split())

PREDETERMINERS: FrozenSet[str] = frozenset("all both half quite rather such".split())

PERSONAL_PRONOUNS: Dict[str, str] = {
    "i": "1sg", "me": "1sg", "my": "1sg", "mine": "1sg", "myself": "1sg",
    "we": "1pl", "us": "1pl", "our": "1pl", "ours": "1pl", "ourselves": "1pl",
    "you": "2", "your": "2", "yours": "2", "yourself": "2sg", "yourselves": "2pl",
    "he": "3sg", "him": "3sg", "his": "3sg", "himself": "3sg",
    "she": "3sg", "her": "3sg", "hers": "3sg", "herself": "3sg",
    "it": "3sg", "its": "3sg", "itself": "3sg",
    "they": "3pl", "them": "3pl", "their": "3pl", "theirs": "3pl",
    "themselves": "3pl",
}

POSSESSIVE_PRONOUNS: FrozenSet[str] = frozenset({
    "my", "your", "his", "her", "its", "our", "their", "whose", "mine", "yours",
    "hers", "ours", "theirs",
})

WH_PRONOUNS: FrozenSet[str] = frozenset({"who", "whom", "whose", "which", "what"})
WH_DETERMINERS: FrozenSet[str] = frozenset({"what", "which", "whose"})
WH_ADVERBS: FrozenSet[str] = frozenset({"when", "where", "why", "how"})

MODALS: FrozenSet[str] = frozenset({
    "can", "could", "may", "might", "must", "shall", "should", "will", "would",
    "ought", "need", "dare", "used",
})

COORDINATING_CONJ: FrozenSet[str] = frozenset({
    "and", "or", "but", "nor", "so", "yet", "for", "plus", "versus", "vs",
})

SUBORDINATING_CONJ: FrozenSet[str] = frozenset({
    "although", "though", "because", "since", "unless", "until", "while",
    "whereas", "if", "when", "whenever", "where", "wherever", "after", "before",
    "as", "that", "whether", "once", "so that", "in order that",
})

PREPOSITIONS: FrozenSet[str] = frozenset("""
about above across after against along among around as at before behind below
beneath beside between beyond by despite down during except for from in inside
into like near of off on onto out outside over past per since than through
throughout to toward towards under underneath unlike until up upon via with
within without according across amid anti atop circa despite unto versus
""".split())

PARTICLES: FrozenSet[str] = frozenset({
    "up", "out", "off", "down", "away", "back", "over", "under", "in", "on",
    "through", "along", "around", "about",
})

INTERJECTIONS: FrozenSet[str] = frozenset({
    "oh", "ah", "wow", "ouch", "oops", "hmm", "huh", "hey", "uh", "um", "er",
    "alas", "bravo", "yikes", "wow", "gee", "oops",
})

# --------------------------------------------------------------------------- #
# Verb lexicon -- the highest-ambiguity class, so a table is worth it
# --------------------------------------------------------------------------- #
VERB_FORMS: Dict[str, Tuple[str, str]] = {
    # base: (3sg, past_participle)
    "be": ("is", "been"), "have": ("has", "had"), "do": ("does", "done"),
    "say": ("says", "said"), "go": ("goes", "gone"), "make": ("makes", "made"),
    "take": ("takes", "taken"), "see": ("sees", "seen"), "know": ("knows", "known"),
    "think": ("thinks", "thought"), "come": ("comes", "come"), "want": ("wants", "wanted"),
    "use": ("uses", "used"), "find": ("finds", "found"), "give": ("gives", "given"),
    "tell": ("tells", "told"), "work": ("works", "worked"), "call": ("calls", "called"),
    "try": ("tries", "tried"), "ask": ("asks", "asked"), "need": ("needs", "needed"),
    "feel": ("feels", "felt"), "become": ("becomes", "become"), "leave": ("leaves", "left"),
    "put": ("puts", "put"), "mean": ("means", "meant"), "keep": ("keeps", "kept"),
    "let": ("lets", "let"), "begin": ("begins", "begun"), "seem": ("seems", "seemed"),
    "help": ("helps", "helped"), "talk": ("talks", "talked"), "turn": ("turns", "turned"),
    "start": ("starts", "started"), "show": ("shows", "shown"), "hear": ("hears", "heard"),
    "run": ("runs", "run"), "move": ("moves", "moved"), "like": ("likes", "liked"),
    "live": ("lives", "lived"), "believe": ("believes", "believed"),
    "hold": ("holds", "held"), "bring": ("brings", "brought"),
    "happen": ("happens", "happened"), "write": ("writes", "written"),
    "provide": ("provides", "provided"), "sit": ("sits", "sat"),
    "stand": ("stands", "stood"), "lose": ("loses", "lost"), "pay": ("pays", "paid"),
    "meet": ("meets", "met"), "include": ("includes", "included"),
    "continue": ("continues", "continued"), "set": ("sets", "set"),
    "learn": ("learns", "learned"), "change": ("changes", "changed"),
    "lead": ("leads", "led"), "understand": ("understands", "understood"),
    "watch": ("watches", "watched"), "follow": ("follows", "followed"),
    "stop": ("stops", "stopped"), "create": ("creates", "created"),
    "speak": ("speaks", "spoken"), "read": ("reads", "read"),
    "allow": ("allows", "allowed"), "add": ("adds", "added"),
    "spend": ("spends", "spent"), "grow": ("grows", "grown"),
    "open": ("opens", "opened"), "walk": ("walks", "walked"),
    "win": ("wins", "won"), "offer": ("offers", "offered"),
    "remember": ("remembers", "remembered"), "love": ("loves", "loved"),
    "consider": ("considers", "considered"), "appear": ("appears", "appeared"),
    "buy": ("buys", "bought"), "wait": ("waits", "waited"),
    "serve": ("serves", "served"), "die": ("dies", "died"),
    "send": ("sends", "sent"), "expect": ("expects", "expected"),
    "build": ("builds", "built"), "stay": ("stays", "stayed"),
    "fall": ("falls", "fallen"), "cut": ("cuts", "cut"),
    "reach": ("reaches", "reached"), "kill": ("kills", "killed"),
    "remain": ("remains", "remained"), "suggest": ("suggests", "suggested"),
    "raise": ("raises", "raised"), "pass": ("passes", "passed"),
    "sell": ("sells", "sold"), "require": ("requires", "required"),
    "report": ("reports", "reported"), "decide": ("decides", "decided"),
    "pull": ("pulls", "pulled"), "explain": ("explains", "explained"),
    "carry": ("carries", "carried"), "develop": ("develops", "developed"),
    "design": ("designs", "designed"), "manage": ("manages", "managed"),
    "deploy": ("deploys", "deployed"), "implement": ("implements", "implemented"),
    "optimise": ("optimises", "optimised"), "optimize": ("optimizes", "optimized"),
    "test": ("tests", "tested"), "debug": ("debugs", "debugged"),
    "review": ("reviews", "reviewed"), "mentor": ("mentors", "mentored"),
    "collaborate": ("collaborates", "collaborated"), "deliver": ("delivers", "delivered"),
    "migrate": ("migrates", "migrated"), "scale": ("scales", "scaled"),
    "measure": ("measures", "measured"), "prioritise": ("prioritises", "prioritised"),
    "prioritize": ("prioritizes", "prioritized"), "troubleshoot": ("troubleshoots", "troubleshot"),
    "document": ("documents", "documented"), "refactor": ("refactors", "refactored"),
    "profile": ("profiles", "profiled"), "benchmark": ("benchmarks", "benchmarked"),
    "containerise": ("containerises", "containerised"), "containerize": ("containerizes", "containerized"),
    "ship": ("ships", "shipped"), "own": ("owns", "owned"), "drive": ("drives", "driven"),
    "influence": ("influences", "influenced"), "negotiate": ("negotiates", "negotiated"),
    "coach": ("coaches", "coached"), "hire": ("hires", "hired"),
    "interview": ("interviews", "interviewed"), "present": ("presents", "presented"),
    "publish": ("publishes", "published"), "research": ("researches", "researched"),
    "analyze": ("analyzes", "analyzed"), "analyse": ("analyses", "analysed"),
}

#: derived inflected forms -> (lemma, tag)
INFLECTED_VERBS: Dict[str, Tuple[str, str]] = {}
for _base, (_sg, _pp) in VERB_FORMS.items():
    INFLECTED_VERBS[_sg] = (_base, TAG_VERB_3SG)
    INFLECTED_VERBS[_base + "ing"] = (_base, TAG_VERB_GERUND)
    INFLECTED_VERBS[_pp] = (_base, TAG_VERB_PAST_PART)
    if _base.endswith("e"):
        INFLECTED_VERBS[_base + "d"] = (_base, TAG_VERB_PAST)
    elif _base.endswith("y") and _base[-2:-1] not in "aeiou":
        INFLECTED_VERBS[_base[:-1] + "ied"] = (_base, TAG_VERB_PAST)
    else:
        INFLECTED_VERBS[_base + "ed"] = (_base, TAG_VERB_PAST)

# --------------------------------------------------------------------------- #
# Adjectives / adverbs / nouns commonly seen in interviews
# --------------------------------------------------------------------------- #
ADJECTIVES: FrozenSet[str] = frozenset("""
good great excellent strong weak best better worse worst bad poor high higher
highest low lower lowest large larger largest small smaller smallest big
bigger biggest fast faster fastest slow slower slowest quick quicker quickest
early earlier earliest late later latest new newer newest old older oldest
young younger youngest long longer longest short shorter shortest hard harder
hardest easy easier easiest simple simpler simplest complex difficult clear
clearer clearest close closer closest deep deeper deepest strong stronger
strongest weak full empty happy happier happiest sad angry nervous anxious
confident curious motivated passionate driven reliable robust scalable
maintainable testable secure efficient accurate relevant useful helpful
technical nontechnical senior junior staff principal lead entry mid level
remote onsite hybrid fulltime parttime contract freelance
""".split())

ADVERBS: FrozenSet[str] = frozenset("""
very really quite rather pretty fairly extremely incredibly highly slightly
somewhat too enough almost nearly just only even still yet already always
never often sometimes rarely usually occasionally generally specifically
particularly especially mainly mostly largely partly recently previously
currently finally eventually immediately quickly slowly carefully clearly
directly effectively successfully independently collaboratively
""".split())

NOUNS: FrozenSet[str] = frozenset("""
time year people way day man thing woman life child world school state family
student group country problem hand part place case week company system program
question work government number night point home water room mother area money
story fact month lot right study book eye job word business issue side kind
head house service friend father power hour game line end member law car city
community name president team minute idea kid body information back parent face
others level office door health person art war history party result change
morning reason research girl guy moment air teacher force education
experience skills skill role roles team teams project projects product products
company companies startup startups engineer engineers engineering manager
managers management leader leadership design designs architecture code codes
software hardware database databases server servers cloud api apis service
services microservice microservices latency throughput availability reliability
scalability security privacy testing tests test coverage deployment deployments
pipeline pipelines monitor monitoring alerting logging metrics dashboards
interview interviewer interviewee candidate candidates recruiter recruiters
offer offers salary compensation equity stock bonus benefits culture values
mission vision strategy roadmap goal goals objective objectives impact
growth learning mentor mentoring feedback performance review promotion
challenge challenges conflict conflicts disagreement priority priorities
tradeoff tradeoffs decision decisions risk risks opportunity opportunities
customer customers user users client clients stakeholder stakeholders
requirement requirements specification spec feature features bug bugs issue
release version branch commit pull request code review refactor migration
algorithm algorithms model models training dataset datasets feature features
accuracy precision recall f1 loss gradient neural network transformer
embedding embeddings vector search index cache queue worker thread process
container containers kubernetes docker terraform aws azure gcp
language languages python java javascript typescript golang rust c cpp sql
nosql redis postgres mysql mongodb kafka rabbitmq spark hadoop
framework frameworks library libraries package packages dependency
documentation documentation wiki runbook postmortem incident outage
""".split())

#: words that are strong topic anchors for interview-domain retrieval
DOMAIN_ANCHORS: FrozenSet[str] = frozenset("""
experience project team leadership conflict failure challenge strength
weakness motivation goal salary culture growth learning impact architecture
scale performance debugging testing deployment ownership collaboration
communication ambiguity deadline pressure priority decision tradeoff
""".split())


def build_pos_lexicon() -> Dict[str, str]:
    """Assemble the unigram ``word -> most likely tag`` lookup table."""
    table: Dict[str, str] = {}
    for word in DETERMINERS:
        table[word] = TAG_DET
    for word in PREDETERMINERS:
        table[word] = TAG_PREDET
    for word in PERSONAL_PRONOUNS:
        table[word] = TAG_POSS_PRONOUN if word in POSSESSIVE_PRONOUNS else TAG_PRONOUN
    for word in MODALS:
        table[word] = TAG_MODAL
    for word in COORDINATING_CONJ:
        table[word] = TAG_CONJ
    for word in SUBORDINATING_CONJ:
        table[word] = TAG_PREP
    for word in PREPOSITIONS:
        table[word] = TAG_PREP
    for word in WH_PRONOUNS:
        table[word] = TAG_WH_PRONOUN
    for word in WH_ADVERBS:
        table[word] = TAG_WH_ADV
    for word in INTERJECTIONS:
        table[word] = TAG_INTERJ
    for word in ADJECTIVES:
        table[word] = TAG_ADJ
    for word in ADVERBS:
        table[word] = TAG_ADV
    for word in NOUNS:
        table[word] = TAG_NOUN_SINGULAR
    for word, (_lemma, tag) in INFLECTED_VERBS.items():
        table.setdefault(word, tag)
    for word in VERB_FORMS:
        table.setdefault(word, TAG_VERB_BASE)
    for word in STOP_WORDS:
        table.setdefault(word, TAG_ADV)
    return table


POS_LEXICON: Dict[str, str] = build_pos_lexicon()

#: be-forms and other high frequency verbs that the table above cannot derive
_BE_FORMS: Dict[str, str] = {
    "am": TAG_VERB_NON3SG, "is": TAG_VERB_3SG, "are": TAG_VERB_NON3SG,
    "was": TAG_VERB_PAST, "were": TAG_VERB_PAST, "been": TAG_VERB_PAST_PART,
    "being": TAG_VERB_GERUND, "be": TAG_VERB_BASE, "'m": TAG_VERB_NON3SG,
    "'re": TAG_VERB_NON3SG, "'s": TAG_VERB_3SG,
}
POS_LEXICON.update(_BE_FORMS)

#: tokens that must never be re-tagged by the contextual rules
_PROTECTED_TAGS = frozenset({
    TAG_DET, TAG_PREDET, TAG_PRONOUN, TAG_POSS_PRONOUN, TAG_MODAL, TAG_TO,
    TAG_WH_PRONOUN, TAG_WH_DET, TAG_WH_ADV, TAG_CONJ, TAG_CARDINAL,
    TAG_PUNCT, TAG_POSSESSIVE, TAG_SYMBOL, TAG_FOREIGN,
})
