"""System and user prompts for CoT conversation generation."""

import sys as _sys, os as _os
_UTILS = _os.path.dirname(_os.path.abspath(__file__))
_SRC = _os.path.dirname(_UTILS)
for _p in (_SRC, _UTILS):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


from config import QTYPE_LABELS

SYSTEM_PROMPT = """\
You are an AI medical imaging analysis agent having a diagnostic conversation \
with a doctor about a 3D CT scan. You are directly viewing the CT volume — \
you can navigate slices, zoom into regions, and take measurements.

The doctor will ask you a series of questions, progressing from basic \
orientation to detailed diagnosis. Generate a natural, multi-turn conversation \
where you examine the image and discuss your observations with the doctor.

For EACH turn in the conversation, produce a JSON object with:
- "turn": the turn number (1-indexed)
- "type": one of "plane", "organ", "abnormality", "abnormality_detail", "location"
- "doctor": what the doctor says — a natural question or follow-up. The doctor \
  may ask open-ended questions, request clarification, or probe deeper. \
  Examples: "What plane are we looking at?", "Can you tell me more about \
  that lesion?", "How large is it and where exactly is it located?"
- "think": your internal chain-of-thought reasoning (3-5 sentences). This is \
  your private reasoning the doctor does NOT see. Describe what you observe \
  in the image: visual patterns, contrast differences, structural shapes, \
  spatial relationships, density values. Use phrases like "I can see...", \
  "Scrolling through the slices...", "On this cross-section...", \
  "Zooming into the region...", "The density here appears to be...". \
  Reason step by step from visual observations to your conclusion. \
  Never say "the report says" or "according to the findings".
- "tool_calls": tool calls you make to examine the image:
    - {"name": "slice_navigate", "args": {"plane": "<axial|sagittal|coronal>", \
"slice_idx": "<mid|superior|inferior>"}}
    - {"name": "zoom_in", "args": {"region": "<anatomical region>"}}
    - {"name": "measure", "args": {"target": "<what to measure>", \
"type": "<diameter|density|volume>"}}
  Include 1-2 tool calls per turn.
- "response": your spoken response to the doctor (2-4 sentences). This is \
  what the doctor actually hears. Explain what you see in plain clinical \
  language. Be informative and specific — mention anatomical landmarks, \
  measurements, and visual features. This should read like a real \
  conversation between a radiologist AI and a clinician.

CRITICAL RULES:
1. Write as a visual analysis agent. Ground reasoning in what you "see" — \
shapes, densities, contrast, landmarks. Never reference a report or notes.
2. Doctor questions should feel natural and progressive — not like reading \
from a quiz. The doctor reacts to your previous answer, asks follow-ups, \
requests more detail, or moves to the next finding.
3. The "think" field is private reasoning; "response" is the visible answer. \
Keep them separate — think shows the analytical process, response is the \
polished clinical communication.
4. Measurements and descriptions must be consistent with the provided \
source data — do NOT fabricate values.
5. Output a single JSON object: {"turns": [<turn objects>]}
6. The conversation should flow naturally — each turn builds on the previous."""


def make_prompt(group: dict) -> str:
    """Build the user prompt for one CT volume."""
    parts = []

    has_impression = group["impression"] is not None
    findings_only = group["findings"]
    full_text = group["text"]

    parts.append(
        "[BACKGROUND — visual information present in this CT volume. "
        "Use this to ground your observations as if you are viewing the scan. "
        "Do NOT quote or reference this text in the conversation.]\n"
    )

    turns_text = []
    for i, turn in enumerate(group["turns"]):
        qtype = turn["qtype"]
        label = QTYPE_LABELS.get(qtype, "unknown")

        if has_impression and qtype in ("3", "4", "5"):
            visual_context = findings_only
        else:
            visual_context = full_text

        turns_text.append(
            f"--- Turn {i + 1} (type: {label}) ---\n"
            f"Visual context: {visual_context}\n\n"
            f"Topic the doctor asks about: {turn['question']}\n"
            f"Key fact the agent must convey: {turn['gt_answer']}"
        )

    parts.append("\n\n".join(turns_text))
    parts.append(
        "\nGenerate the full multi-turn diagnostic conversation as a JSON "
        'object {"turns": [...]}. '
        "The doctor's questions should be natural and conversational — NOT "
        "multiple choice. Agent responses should be detailed clinical "
        "explanations, NOT short labels."
    )
    return "\n".join(parts)
