"""
explainer.py
────────────
LLM reasoning layer with triple fallback:
  1. Groq     (free, fastest — llama3)
  2. Gemini   (free, good quality)
  3. Claude   (paid, best quality)
  4. Template (always works, no API needed)

Add to .env whichever you have:
  GROQ_API_KEY=...       free → console.groq.com
  GEMINI_API_KEY=...     free → aistudio.google.com/apikey
  ANTHROPIC_API_KEY=...  paid → console.anthropic.com
"""

import os
import json
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")

CLASS_NAMES = [
    "Dense_Vegetation", "Dry_Vegetation", "Ground_Objects",
    "Rocks", "Landscape", "Sky"
]
TRAVERSABILITY = {
    "Dense_Vegetation": 0.45,
    "Dry_Vegetation":   0.75,
    "Ground_Objects":   0.30,
    "Rocks":            0.15,
    "Landscape":        0.95,
    "Sky":              0.00,
}


# ─────────────────────────────────────────────
# UNIFIED LLM CALLER
# ─────────────────────────────────────────────

def _call_llm(prompt: str, max_tokens: int = 200):
    """
    Try Groq → Gemini → Claude → None.
    Returns text or None.
    """
    # 1. Groq
    if GROQ_API_KEY:
        try:
            from groq import Groq
            client   = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            print("  [LLM] Groq ✅")
            return response.choices[0].message.content
        except Exception as e:
            print(f"  [LLM] Groq failed: {e}")

    # 2. Gemini
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model    = genai.GenerativeModel("gemini-2.0-flash-lite")
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.3,
                )
            )
            print("  [LLM] Gemini ✅")
            return response.text
        except Exception as e:
            print(f"  [LLM] Gemini failed: {e}")

    # 3. Claude
    if ANTHROPIC_API_KEY:
        try:
            from anthropic import Anthropic
            client   = Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            print("  [LLM] Claude ✅")
            return response.content[0].text
        except Exception as e:
            print(f"  [LLM] Claude failed: {e}")

    print("  [LLM] All APIs failed — using template")
    return None


# ─────────────────────────────────────────────
# TEMPLATE FALLBACKS (no API needed)
# ─────────────────────────────────────────────

def _template_briefing(class_dist, path_result,
                        risk_zones, similar_terrains,
                        avg_traversability) -> str:
    dominant  = max(class_dist, key=class_dist.get)
    dom_pct   = class_dist[dominant] * 100
    hops      = path_result.get("hop_count", 0)
    n_similar = len(similar_terrains)
    trav_pct  = avg_traversability * 100

    risk_names = []
    for z in risk_zones[:3]:
        attr = z.get("attributes", {})
        cls  = attr.get("class_name", "")
        sev  = attr.get("severity", "")
        if cls:
            risk_names.append(f"{cls} ({sev})")

    risk_str = ", ".join(risk_names) if risk_names \
               else "none detected"
    mem_str  = (f"TigerGraph memory matched {n_similar} "
                f"similar past terrains. ") \
               if n_similar > 0 else ""

    return (
        f"Scene is {dom_pct:.0f}% "
        f"{dominant.replace('_',' ')} with average "
        f"traversability {trav_pct:.0f}%. TigerGraph "
        f"Dijkstra identified a {hops}-hop safe path. "
        f"Risk zones: {risk_str}. {mem_str}"
        f"Recommendation: proceed via Landscape and "
        f"Dry Vegetation, avoid HIGH risk zones."
    )


def _template_explanation(class_dist, path_result,
                           risk_zones, model_miou) -> str:
    dominant = max(class_dist, key=class_dist.get)
    hops     = path_result.get("hop_count", 0)
    n_risks  = len(risk_zones)
    return (
        f"This system fuses DeepLabV3+ segmentation "
        f"(mIoU={model_miou:.4f}) with TigerGraph spatial "
        f"reasoning for autonomous desert navigation. The "
        f"{dominant.replace('_',' ')}-dominated scene was "
        f"decomposed into a traversability graph, with "
        f"{n_risks} risk zones avoided via Dijkstra "
        f"over {hops} nodes — uniquely combining CV, "
        f"graph databases, and LLM reasoning."
    )


# ─────────────────────────────────────────────
# PUBLIC FUNCTIONS
# ─────────────────────────────────────────────

def generate_navigation_briefing(
    class_dist:         dict,
    path_result:        dict,
    risk_zones:         list,
    similar_terrains:   list,
    avg_traversability: float,
) -> str:
    context = {
        "scene_composition": {
            k: f"{v*100:.1f}%"
            for k, v in class_dist.items() if v > 0.01
        },
        "traversability_scores": {
            k: TRAVERSABILITY.get(k, 0) for k in class_dist
        },
        "average_traversability": round(avg_traversability, 3),
        "safest_path": {
            "hops":   path_result.get("hop_count", 0),
            "cost":   round(path_result.get("total_cost", 0), 3),
            "method": path_result.get("note", "Dijkstra"),
        },
        "risk_zones": [
            {
                "class":    z.get("attributes", {})
                             .get("class_name", "Unknown"),
                "severity": z.get("attributes", {})
                             .get("severity", "?"),
                "patches":  z.get("attributes", {})
                             .get("patch_count", 0),
            }
            for z in risk_zones[:5]
        ],
        "similar_terrains_in_memory": len(similar_terrains),
        "historical_avg_trav": round(
            sum(t.get("attributes", {})
                 .get("avg_traversability", 0)
                for t in similar_terrains)
            / max(len(similar_terrains), 1), 3
        ) if similar_terrains else None,
    }

    prompt = f"""You are an AI terrain analyst for an autonomous \
offroad vehicle (UGV) in desert terrain.

Based ONLY on this TigerGraph data, write a navigation briefing \
(max 100 words, flowing prose, no bullet points):

{json.dumps(context, indent=2)}

Rules:
- Only use data above, no assumptions
- Name HIGH/MED_HIGH risk zones explicitly
- State recommended path clearly
- Mention historical data if present
- End with one actionable recommendation"""

    result = _call_llm(prompt, max_tokens=180)
    return result if result else _template_briefing(
        class_dist, path_result,
        risk_zones, similar_terrains,
        avg_traversability,
    )


def generate_system_explanation(
    class_dist:  dict,
    path_result: dict,
    risk_zones:  list,
    model_miou:  float,
) -> str:
    dominant = max(class_dist, key=class_dist.get)
    n_risks  = len(risk_zones)
    hops     = path_result.get("hop_count", 0)

    prompt = f"""You are pitching an AI terrain intelligence \
system to hackathon judges. Write ONE paragraph (60-80 words), \
confident, technical but accessible.

System results:
- Dominant terrain: {dominant} \
({class_dist.get(dominant,0)*100:.1f}%)
- Risk zones found: {n_risks}
- Safe path: {hops} hops via TigerGraph Dijkstra
- Model mIoU: {model_miou:.4f}
- Stack: DeepLabV3+ → TigerGraph → LLM reasoning

Explain what happened, why it matters for autonomous \
vehicles, and what makes this approach unique."""

    result = _call_llm(prompt, max_tokens=130)
    return result if result else _template_explanation(
        class_dist, path_result, risk_zones, model_miou)


def generate_failure_analysis(
    low_iou_classes: list,
    confusion_pairs: list,
) -> str:
    prompt = f"""AI model analyst. In 60-80 words explain:
1. Why these desert classes are hard to segment
2. What navigation risk this creates
3. One improvement suggestion

Weak classes: {json.dumps(low_iou_classes)}
Confused pairs: {json.dumps(confusion_pairs)}
Only use provided data. Be specific."""

    result = _call_llm(prompt, max_tokens=130)
    if result:
        return result

    weak = ", ".join(low_iou_classes[:2]) \
           if low_iou_classes else "some classes"
    return (
        f"{weak} are challenging due to visual similarity "
        f"in desert environments, creating navigation risk "
        f"when the UGV misidentifies high-risk zones. "
        f"Recommendation: increase augmentation diversity "
        f"for these classes in the training pipeline."
    )


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("="*50)
    print("API Keys status:")
    print(f"  Groq      : {'✅ found' if GROQ_API_KEY else '❌ not set'}")
    print(f"  Gemini    : {'✅ found' if GEMINI_API_KEY else '❌ not set'}")
    print(f"  Anthropic : {'✅ found' if ANTHROPIC_API_KEY else '❌ not set'}")
    print("="*50)

    test_dist  = {
        "Dense_Vegetation": 0.12, "Dry_Vegetation": 0.28,
        "Ground_Objects": 0.06,   "Rocks": 0.11,
        "Landscape": 0.41,        "Sky": 0.02,
    }
    test_path  = {"hop_count": 14, "total_cost": 1.23}
    test_risks = [
        {"attributes": {"class_name": "Rocks",
                        "severity": "HIGH", "patch_count": 5}},
        {"attributes": {"class_name": "Ground_Objects",
                        "severity": "MED_HIGH", "patch_count": 3}},
    ]
    test_sim   = [
        {"attributes": {"avg_traversability": 0.71,
                        "dominant_class": "Landscape",
                        "miou_achieved": 0.68}},
    ]

    print("\nTesting briefing...")
    b = generate_navigation_briefing(
        test_dist, test_path, test_risks, test_sim, 0.63)
    print(f"\n📋 Briefing:\n{b}")

    print("\nTesting explanation...")
    e = generate_system_explanation(
        test_dist, test_path, test_risks, 0.6676)
    print(f"\n🧠 Explanation:\n{e}")