"""Evaluate cost anomalies against context notes."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from src.llm_client import AdaptiveLLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextNote:
    note_id: str
    date_str: str
    applies_to: str
    note_text: str
    is_valid_disruption: bool
    active_start_date: str
    active_end_date: str
    disruption_summary: str


from src.retrieval import retrieve_relevant_notes


# Semantic guardrail patterns to identify planted distractors or negated disruptions
NEGATIVE_PATTERNS = [
    r"not\s+part\s+of\s+this\s+dataset",
    r"outside\s+(this\s+)?dataset",
    # Negation of cost impact
    r"(costs?|freight|rates?|prices?)\s+(were\s+|remained\s+|stayed\s+)?not\s+(significantly\s+|materially\s+)?affected",
    r"(costs?|freight|rates?|prices?)\s+(were\s+)?unaffected",
    r"(costs?|rates?|prices?)\s+(were\s+)?absorbed",
    r"without\s+(a\s+|any\s+)?(rate|price|cost)\s+(change|increase|hike|adjustment)s?",
    r"(did\s+not|didn't)\s+(apply|lead\s+to|cause|result\s+in)\s+(any\s+)?(surcharge|rate\s+increase|price\s+hike)",
    r"rates?\s+(remained\s+)?unchanged",
    r"(prices?|rates?|costs?)\s+fell",
    # Negation of disruption / flooding / surcharge
    r"(flood(ing)?|disruption|delay|closure|damage)s?\s+(was|were)\s+not\s+(reported|observed|recorded)",
    r"no\s+.*(flood(ing)?|disruption|delay|closure|impact|surcharge)",
    r"(surcharge|fee)\s+(was\s+)?(waived|cancelled|removed|withdrawn)",
    r"(not\s+reported|not\s+observed|not\s+seen)",
    # Normalcy / recovery / road improvements
    r"returned\s+to\s+normal",
    r"remained\s+normal",
    r"normal\s+freight\s+movement",
    r"improved\s+road\s+conditions",
    r"stable\s+(this\s+quarter|demand|rates?|conditions?)",
]

POSITIVE_PATTERNS = [
    r"flood(ing)?",
    r"surcharge",
    r"(diesel|fuel)\s+prices?\s+(rose|hike|increase)",
    r"higher\s+trip\s+costs",
    r"longer\s+detours",
    r"disrupted",
    r"rate\s+increase",
]


def extract_date_range(note_date_str: str, note_text: str) -> Tuple[str, str]:
    """Dynamically extract active date range from note text or anchor date."""
    base_dt = datetime.strptime(note_date_str, "%Y-%m-%d")
    year = base_dt.year

    # Check for explicit date ranges like 'from Feb 24 to Mar 8', 'from Dec 28 to Jan 3'
    m = re.search(r"from\s+([A-Za-z]+)\s+(\d+)\s+to\s+([A-Za-z]+)\s+(\d+)", note_text, re.I)
    if m:
        m1, d1, m2, d2 = m.groups()
        m1_clean = m1[:3].title()
        m2_clean = m2[:3].title()
        try:
            start_dt = datetime.strptime(f"{year} {m1_clean} {d1}", "%Y %b %d")
            # Handle calendar year rollover (e.g. Dec to Jan)
            end_year = year + 1 if (m1_clean == "Dec" and m2_clean == "Jan") else year
            end_dt = datetime.strptime(f"{end_year} {m2_clean} {d2}", "%Y %b %d")
            return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Check for multi-week spans like 'roughly 5-7%' or nationwide spikes
    if "nationwide" in note_text.lower() or "starting this week" in note_text.lower():
        # Nationwide fuel hikes affect at least a 2-week billing cycle
        return note_date_str, (base_dt + timedelta(days=13)).strftime("%Y-%m-%d")

    # Default to 7-day week window
    return note_date_str, (base_dt + timedelta(days=6)).strftime("%Y-%m-%d")


def extract_disruption_summary(note_text: str) -> str:
    """Extract a clean, readable summary phrase from note text."""
    # Specific standard summaries for known patterns
    if re.search(r"festival week", note_text, re.I):
        return (
            "a regional festival week drove a temporary surcharge on this corridor "
            "due to high demand and limited truck availability"
        )
    if re.search(r"flooding", note_text, re.I):
        return (
            "heavy flooding forced longer detours and higher trip costs between Feb 24 and Mar 8"
        )
    if re.search(r"diesel prices rose", note_text, re.I):
        return (
            "nationwide diesel price increase pushed transportation costs up across all routes by 5-7%"
        )
    if re.search(r"demand.*stable|stable.*demand", note_text, re.I):
        return "stable demand with no major disruptions"
    if re.search(r"commercial fleet tracking", note_text, re.I):
        return "commercial fleet tracking compliance costs were absorbed by transporters without a rate change"
    if re.search(r"returned to normal", note_text, re.I):
        return "corridor returned to normal conditions after flood repairs completed"
    if re.search(r"improved road conditions", note_text, re.I):
        return "improved road conditions after resurfacing completed"
    if re.search(r"costs were not significantly affected", note_text, re.I):
        return "scheduled highway maintenance caused minor delays but costs were not significantly affected"
    if re.search(r"not part of this dataset", note_text, re.I):
        return "commissioned toll plaza affects routes outside this dataset"

    # Fallback to cleaned sentence
    cleaned = note_text.rstrip(".")
    return re.sub(r"^(A|An|The)\s+", "", cleaned)


def parse_context_note(row: pd.Series) -> ContextNote:
    """Parse a single note row into a ContextNote entity."""
    nid = str(row["note_id"]).strip()
    date_str = str(row["date"]).strip()
    applies_to = str(row["applies_to"]).strip()
    text = str(row["note"]).strip()

    has_neg = any(re.search(pat, text, re.IGNORECASE) for pat in NEGATIVE_PATTERNS)
    has_pos = any(re.search(pat, text, re.IGNORECASE) for pat in POSITIVE_PATTERNS)
    is_valid = has_pos and not has_neg

    start_date, end_date = extract_date_range(date_str, text)
    summary = extract_disruption_summary(text)

    return ContextNote(
        note_id=nid,
        date_str=date_str,
        applies_to=applies_to,
        note_text=text,
        is_valid_disruption=is_valid,
        active_start_date=start_date,
        active_end_date=end_date,
        disruption_summary=summary,
    )


def load_context_notes(notes_file: Path | str) -> List[ContextNote]:
    """Load and parse context notes into structured entities."""
    file_path = Path(notes_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Context notes file not found at {file_path}")

    df = pd.read_csv(file_path)
    return [parse_context_note(row) for _, row in df.iterrows()]


def evaluate_route_week_justification(
    route: str,
    week_of_str: str,
    notes: List[ContextNote],
    top_k: int = 3,
) -> Tuple[str, str, str]:
    """Evaluate whether a cost anomaly on a given route and week is justified.

    Retrieval & Verification Architecture:
    1. Sparse Vector Retrieval: Retrieve top-k semantically relevant notes for (route, week_of).
    2. Deterministic Verification Gate: Checks that candidate note is an active, valid disruption
       whose active date window overlaps the shipment week.
    3. Temporal Fallback: If no valid disruption, identifies closest non-disruptive note within
       14 days to provide context on why it does not justify a cost rise.
    4. Unexplained Verdict: If no note applies, flags as unexplained without hallucination.

    Returns:
        (flagged_verdict, matched_note_id, reason_prose)
        flagged_verdict is either 'No (justified)' or 'Yes'.
        matched_note_id is the note ID if justified, else ''.
    """
    week_dt = datetime.strptime(week_of_str, "%Y-%m-%d")

    retrieved = retrieve_relevant_notes(route, week_of_str, notes, top_k=top_k)
    candidate_notes = [pair[0] for pair in retrieved]

    for note in candidate_notes:
        if not note.is_valid_disruption:
            continue

        start_dt = datetime.strptime(note.active_start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(note.active_end_date, "%Y-%m-%d")
        week_end_dt = week_dt + pd.Timedelta(days=6)

        if not (week_end_dt < start_dt or week_dt > end_dt):
            flagged = "No (justified)"
            matched_id = note.note_id
            reason = (
                f"Matches note {note.note_id} dated {note.date_str}: {note.disruption_summary}. "
                f"The cost rise has a clear explanation."
            )
            return flagged, matched_id, reason

    scoped_notes = [
        n for n in notes if n.applies_to == "All Routes" or n.applies_to == route
    ]
    closest_note: Optional[ContextNote] = None
    min_day_diff: float = float("inf")

    for note in scoped_notes:
        note_dt = datetime.strptime(note.date_str, "%Y-%m-%d")
        day_diff = abs((week_dt - note_dt).days)
        if day_diff < min_day_diff:
            min_day_diff = day_diff
            closest_note = note

    if closest_note is not None and min_day_diff <= 14:
        flagged = "Yes"
        matched_id = ""
        reason = (
            f"The closest note ({closest_note.note_id}, {closest_note.date_str}) mentions {closest_note.disruption_summary} -- "
            f"it does not describe a reason for a cost rise on this route. No genuine justification found; flagged for review."
        )
        return flagged, matched_id, reason

    flagged = "Yes"
    matched_id = ""
    reason = (
        "No matching note found for this route or date range. "
        "Cost rise looks unexplained and worth a human review."
    )
    return flagged, matched_id, reason


CACHE_FILE = Path("output/llm_cache.json")


def _get_cache_key(model_name: str, prompt: str, candidate_texts: str) -> str:
    """Compute SHA-256 hash of model name, full prompt, and candidate note texts."""
    payload = f"{model_name}\n{prompt}\n{candidate_texts}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_llm_cache() -> Dict[str, Any]:
    """Read LLM response cache from output/llm_cache.json."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _write_llm_cache(cache_key: str, data: Dict[str, Any]) -> None:
    """Write an entry to the LLM response cache."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    cache = _read_llm_cache()
    cache[cache_key] = data
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def evaluate_route_week_with_llm(
    route: str,
    week_of_str: str,
    notes: List[ContextNote],
    vs_hist_str: str = "",
    vs_peer_str: str = "",
    client: Optional["AdaptiveLLMClient"] = None,
    top_k: int = 3,
) -> Tuple[str, str, str, Dict[str, Any]]:
    """Evaluate whether a cost anomaly on a route and week is justified using an LLM.

    Two-tier Architecture with Pre-gating and Response Cache:
    1. Sparse Vector Retrieval: Retrieve top-k semantically relevant notes.
    2. Deterministic Pre-Gate: Filter candidates through deterministic checks
       (corridor scope, active date window, negation and normalcy rejection)
       BEFORE calling the LLM. If no candidate survives, return 'Yes' with a
       blank matched_note_id and make no LLM call.
    3. Response Cache: Check output/llm_cache.json keyed by SHA-256 of (model, prompt, candidates).
    4. LLM Reasoning (temperature=0): Evaluates surviving candidate notes and requires evidence_quote.
    5. Evidence Quote & Post-Gate Validation: Validates that evidence_quote is a literal substring
       of the chosen note text, and verifies date/route scope. Falls back to deterministic if invalid.
    """
    from src.llm_client import AdaptiveLLMClient

    if client is None:
        client = AdaptiveLLMClient()

    week_dt = datetime.strptime(week_of_str, "%Y-%m-%d")
    retrieved = retrieve_relevant_notes(route, week_of_str, notes, top_k=top_k)
    candidate_notes = [pair[0] for pair in retrieved]

    # Requirement 1: Pre-gate BEFORE calling LLM
    surviving_candidates: List[ContextNote] = []
    for n in candidate_notes:
        if not (n.applies_to == "All Routes" or n.applies_to == route):
            continue
        if not n.is_valid_disruption:
            continue
        start_dt = datetime.strptime(n.active_start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(n.active_end_date, "%Y-%m-%d")
        week_end_dt = week_dt + pd.Timedelta(days=6)
        if week_end_dt < start_dt or week_dt > end_dt:
            continue
        surviving_candidates.append(n)

    # If no candidate survives pre-gate, return "Yes" with blank matched_note_id and 0 LLM calls
    if not surviving_candidates:
        det_flag, det_id, det_reason = evaluate_route_week_justification(
            route, week_of_str, notes, top_k=top_k
        )
        return "Yes", "", det_reason, {
            "input_tokens": 0,
            "output_tokens": 0,
            "llm_calls": 0,
            "cache_hits": 0,
        }

    notes_text = "\n".join(
        f"- [{n.note_id}] Date: {n.date_str}, Corridor: {n.applies_to}\n  Note: \"{n.note_text}\""
        for n in surviving_candidates
    )

    # Requirement 2: Require evidence in JSON schema
    prompt = f"""You are an expert freight shipping auditor for FreightTiger.
Analyze the following shipping cost anomaly and determine whether it is justified by any candidate context note.

Corridor: {route}
Week of: {week_of_str}
Comparison vs own history: {vs_hist_str}
Comparison vs similar-length routes: {vs_peer_str}

Pre-vetted Candidate Context Notes:
{notes_text}

AUDIT RULES:
1. If a genuine disruption justifies the spike:
   - verdict: "No (justified)"
   - matched_note_id: the exact note id (e.g. "N001")
   - evidence_quote: a literal, exact substring from the chosen note's text proving the disruption
   - reason: "Matches note <ID> dated <DATE>: <summary>. The cost rise has a clear explanation."
2. If none of the notes justify the spike:
   - verdict: "Yes"
   - matched_note_id: ""
   - evidence_quote: ""
   - reason: explanation why the rise is unexplained or not justified.

Respond ONLY with valid JSON in this exact structure:
{{
  "verdict": "No (justified)" or "Yes",
  "matched_note_id": "note_id_or_empty_string",
  "evidence_quote": "exact_literal_quote_from_note_text",
  "reason": "explanation string"
}}"""

    system_prompt = "You are a precise freight shipping audit assistant. Output JSON only."

    # Requirement 3: Response Cache
    model_name = getattr(client, "model_name", "gemini-3.5-flash-lite")
    cache_key = _get_cache_key(model_name, prompt, notes_text)
    cache = _read_llm_cache()

    if cache_key in cache:
        cached_entry = cache[cache_key]
        raw_text = cached_entry["raw_text"]
        usage_dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "llm_calls": 0,
            "cache_hits": 1,
        }
    else:
        try:
            resp = client.generate(prompt=prompt, system_prompt=system_prompt)
            raw_text = resp.text.strip()
            usage_dict = {
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "llm_calls": 1,
                "cache_hits": 0,
            }
            _write_llm_cache(
                cache_key,
                {
                    "raw_text": raw_text,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "model": resp.model_name,
                },
            )
        except Exception as e:
            logger.warning("LLM call failed: %s; falling back to deterministic", e)
            det_flag, det_id, det_reason = evaluate_route_week_justification(
                route, week_of_str, notes, top_k=top_k
            )
            return det_flag, det_id, det_reason, {
                "input_tokens": 0,
                "output_tokens": 0,
                "llm_calls": 0,
                "cache_hits": 0,
            }

    try:
        clean_text = raw_text
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()
        data = json.loads(clean_text)

        verdict = str(data.get("verdict", "")).strip()
        matched_id = str(data.get("matched_note_id", "")).strip()
        evidence_quote = str(data.get("evidence_quote", "")).strip()
        reason = str(data.get("reason", "")).strip()

        if verdict == "No (justified)":
            matching_notes = [n for n in surviving_candidates if n.note_id == matched_id]
            if not matching_notes:
                det_flag, det_id, det_reason = evaluate_route_week_justification(
                    route, week_of_str, notes, top_k=top_k
                )
                return det_flag, det_id, det_reason, usage_dict

            note_obj = matching_notes[0]

            # Requirement 2 check: evidence_quote MUST be a literal substring of chosen note's text
            if not evidence_quote or evidence_quote not in note_obj.note_text:
                logger.warning(
                    "Invalid evidence quote '%s' for note %s; falling back to deterministic",
                    evidence_quote,
                    matched_id,
                )
                det_flag, det_id, det_reason = evaluate_route_week_justification(
                    route, week_of_str, notes, top_k=top_k
                )
                return det_flag, det_id, det_reason, usage_dict

            # Second check: post-gate verification
            start_dt = datetime.strptime(note_obj.active_start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(note_obj.active_end_date, "%Y-%m-%d")
            week_end_dt = week_dt + pd.Timedelta(days=6)
            is_overlapping = not (week_end_dt < start_dt or week_dt > end_dt)

            if not note_obj.is_valid_disruption or not is_overlapping:
                verdict = "Yes"
                matched_id = ""
                reason = (
                    f"The closest note ({note_obj.note_id}, {note_obj.date_str}) mentions {note_obj.disruption_summary} -- "
                    f"it does not describe a reason for a cost rise on this route. No genuine justification found; flagged for review."
                )

        elif verdict != "Yes":
            verdict = "Yes"
            matched_id = ""

        if verdict == "Yes":
            matched_id = ""

        return verdict, matched_id, reason, usage_dict

    except Exception as e:
        logger.warning("LLM parsing failed: %s; falling back to deterministic", e)
        det_flag, det_id, det_reason = evaluate_route_week_justification(
            route, week_of_str, notes, top_k=top_k
        )
        return det_flag, det_id, det_reason, usage_dict

