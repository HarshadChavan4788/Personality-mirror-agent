import os
import json
from typing import List, Dict, Any

# Try to use crewai library if available
try:
    import crewai
    HAVE_CREWAI = True
except Exception:
    HAVE_CREWAI = False

from generator import call_gemini_api

# -------------------------
# Prompt templates per agent
# -------------------------

TRAIT_PROMPT = """You are Trait Extractor Agent.
Given these user answers (short), extract 6-8 personality traits and give each a numeric score 0-100.
Return STRICT JSON: {{ "traits": {{ "TraitName": score, ... }} }}.
Answers:
{answers}
"""

SUMMARY_PROMPT = """You are Summary Agent.
Given user answers and extracted traits, produce a concise 3-4 sentence personality summary.
Return STRICT JSON: {{ "summary": "..." }}.
Answers:
{answers}
Traits (json):
{traits_json}
"""

VALIDATOR_PROMPT = """You are Validator & Recommendations Agent.
Given the answers and traits produce:
1) 3 practical, concrete recommendations (bullet list).
2) One validating single-sentence encouraging message.
Return STRICT JSON: {{ "recommendations": [...], "validating_message": "..." }}.
Answers:
{answers}
Traits:
{traits_json}
"""

# -------------------------
# CrewAI helpers (best-effort)
# -------------------------

def crew_generate_with_agent(name: str, prompt: str, max_tokens: int = 500) -> str:
    """
    Best-effort wrapper for CrewAI. If crewai is present, attempt to make an agent call.
    The precise crewai SDK may differ; this wrapper tries a few plausible calls.
    If crewai is missing or agent call fails, fallback to call_gemini_api via HTTP.
    """
    if HAVE_CREWAI:
        try:
            # Try a few likely APIs - best-effort (may need adjustment to your crewai SDK)
            # 1) simple generate
            if hasattr(crewai, "generate"):
                resp = crewai.generate(prompt=prompt, max_tokens=max_tokens)
                if isinstance(resp, dict):
                    text = resp.get("text") or resp.get("output") or json.dumps(resp)
                else:
                    text = str(resp)
                return text
            # 2) Agent factory style
            if hasattr(crewai, "Agent"):
                agent = crewai.Agent(name=name, prompt=prompt)
                out = agent.run()
                return out.get("text", str(out))
        except Exception:
            # swallow and fallback
            pass

    # Fallback to Gemini HTTP
    return call_gemini_api(prompt, max_output_tokens=max_tokens)

# -------------------------
# Parse helpers
# -------------------------

def safe_json_parse(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        # Attempt to extract JSON substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            sub = text[start:end+1]
            try:
                return json.loads(sub)
            except Exception:
                pass
    return None

# -------------------------
# Agent runner functions
# -------------------------

def run_trait_agent(answers: List[str]) -> Dict[str, float]:
    prompt = TRAIT_PROMPT.format(answers="\n".join(f"{i+1}. {a}" for i, a in enumerate(answers)))
    raw = crew_generate_with_agent("trait_agent", prompt)
    parsed = safe_json_parse(raw)
    traits = {}
    if parsed and isinstance(parsed, dict) and "traits" in parsed:
        # Normalize scores
        for k, v in parsed["traits"].items():
            try:
                score = float(v)
            except:
                try:
                    score = float(str(v).strip().rstrip("%"))
                except:
                    score = 0.0
            traits[k] = round(max(0.0, min(100.0, score)), 1)
    else:
        # Best-effort fallback: ask simple heuristic extraction if JSON not found
        # We'll ask again with a stricter prompt
        stricter = TRAIT_PROMPT + "\nImportant: Return only JSON and nothing else."
        raw2 = crew_generate_with_agent("trait_agent", stricter)
        parsed2 = safe_json_parse(raw2)
        if parsed2 and "traits" in parsed2:
            for k, v in parsed2["traits"].items():
                try:
                    score = float(v)
                except:
                    score = 0.0
                traits[k] = round(max(0.0, min(100.0, score)), 1)
    return traits

def run_summary_agent(answers: List[str], traits: Dict[str, float]) -> str:
    traits_json = json.dumps(traits)
    prompt = SUMMARY_PROMPT.format(answers="\n".join(f"{i+1}. {a}" for i, a in enumerate(answers)), traits_json=traits_json)
    raw = crew_generate_with_agent("summary_agent", prompt)
    parsed = safe_json_parse(raw)
    if parsed and isinstance(parsed, dict) and "summary" in parsed:
        return parsed["summary"]
    # fallback: return first 300 chars
    return raw.strip()[:1000]

def run_validator_agent(answers: List[str], traits: Dict[str, float]) -> Dict[str, Any]:
    traits_json = json.dumps(traits)
    prompt = VALIDATOR_PROMPT.format(answers="\n".join(f"{i+1}. {a}" for i, a in enumerate(answers)), traits_json=traits_json)
    raw = crew_generate_with_agent("validator_agent", prompt)
    parsed = safe_json_parse(raw)
    out = {"recommendations": [], "validating_message": ""}
    if parsed:
        out["recommendations"] = parsed.get("recommendations", []) if isinstance(parsed.get("recommendations", []), list) else []
        out["validating_message"] = parsed.get("validating_message", "") or parsed.get("validation", "")
    else:
        # Fallback: try to parse lines as recommendations and the last line as validating message
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        recs = []
        val = ""
        if lines:
            # take up to 3 recommendation-like lines starting with - or numbered
            for l in lines:
                if l.startswith("-") or l[0].isdigit():
                    recs.append(l.lstrip("-0123456789. ").strip())
            if not recs:
                # fallback take first 3 lines
                recs = lines[:3]
            val = lines[-1] if len(lines) > 0 else ""
        out["recommendations"] = recs
        out["validating_message"] = val
    return out

# -------------------------
# Orchestrator
# -------------------------

def run_multiagent_personality_pipeline(answers: List[str], name: str = "") -> Dict[str, Any]:
    """
    High-level orchestrator that runs the trait extractor, summary agent, and validator agent.
    Uses crewai if available; otherwise uses Gemini HTTP client in generator.py.
    Returns final composed dictionary with summary, traits, recommendations, validating_message, raw.
    """
    # 1) trait extraction
    traits = run_trait_agent(answers)

    # 2) summary generation
    summary = run_summary_agent(answers, traits)

    # 3) validator & recommendations
    validator_out = run_validator_agent(answers, traits)

    result = {
        "summary": summary,
        "traits": traits,
        "recommendations": validator_out.get("recommendations", []),
        "validating_message": validator_out.get("validating_message", ""),
        "raw": {
            "trait_agent": None,
            "summary_agent": None,
            "validator_agent": None
        }
    }

    # We cannot easily attach raw agent outputs in every environment (crewai shape varies),
    # but we can attempt to store a minimal raw summary by re-running agents with "raw-only" tag:
    try:
        # request raw outputs (non-JSON) for debugging — best-effort
        result["raw"]["trait_agent"] = crew_generate_with_agent("trait_agent_raw", TRAIT_PROMPT.format(answers="\n".join(f"{i+1}. {a}" for i, a in enumerate(answers))))
        result["raw"]["summary_agent"] = crew_generate_with_agent("summary_agent_raw", SUMMARY_PROMPT.format(answers="\n".join(f"{i+1}. {a}" for i, a in enumerate(answers)), traits_json=json.dumps(traits)))
        result["raw"]["validator_agent"] = crew_generate_with_agent("validator_agent_raw", VALIDATOR_PROMPT.format(answers="\n".join(f"{i+1}. {a}" for i, a in enumerate(answers)), traits_json=json.dumps(traits)))
    except Exception:
        # ignore raw fetch failures
        pass

    return result