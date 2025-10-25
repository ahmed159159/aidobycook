import os
import json
import requests
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from urllib.parse import urlencode

# load .env locally
load_dotenv()

# ---------- Config / env ----------
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "").strip()
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/ahmed159/deployedModels/dobby-unhinged-llama-3-3-70b-new-vdw6j81e")
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "").strip()

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
SPOON_BASE = "https://api.spoonacular.com"

# ---------- Helpers ----------
def call_fireworks_chat(system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.2):
    """Call Fireworks chat completions on your deployed model. Returns assistant text or raises."""
    if not FIREWORKS_API_KEY:
        raise RuntimeError("FIREWORKS_API_KEY not set in environment")

    payload = {
        "model": FIREWORKS_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIREWORKS_API_KEY}"
    }
    r = requests.post(FIREWORKS_URL, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Fireworks API error {r.status_code}: {r.text}")
    j = r.json()
    if "choices" in j and len(j["choices"]) > 0:
        choice = j["choices"][0]
        if isinstance(choice.get("message"), dict) and choice["message"].get("content"):
            return choice["message"]["content"].strip()
        if choice.get("text"):
            return choice["text"].strip()
    raise RuntimeError("Fireworks returned no text")


ANALYSIS_SYSTEM_PROMPT = """
You are Dobby, a friendly cooking assistant. Your job: analyze the user's message and return ONLY a compact JSON object (no other text) with the following fields:
- intent: one of "find_recipe", "specific_recipe", "modify_recipe", or "general"
- ingredients: an array of ingredient words (lowercase)
- dish: a short dish name if specific
- exclude: an array of excluded ingredients
- message: short summary in English
If not about cooking, return {"intent":"general","message":"...","ingredients":[],"dish":null,"exclude":[]}
Return JSON only.
"""

def analyze_user_text(user_text: str):
    raw = call_fireworks_chat(ANALYSIS_SYSTEM_PROMPT, user_text, max_tokens=256, temperature=0.1)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[-1].strip()
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    json_text = cleaned[first_brace:last_brace+1] if first_brace != -1 and last_brace != -1 else cleaned
    try:
        parsed = json.loads(json_text)
    except Exception:
        return {"intent": "find_recipe", "ingredients": [user_text], "dish": None, "exclude": [], "message": f"Searching recipes for: {user_text}"}
    parsed.setdefault("ingredients", [])
    parsed.setdefault("exclude", [])
    parsed.setdefault("dish", None)
    parsed.setdefault("message", "")
    parsed.setdefault("intent", "find_recipe")
    return parsed

def spoon_search_by_ingredients(ingredients, exclude=None, number=4):
    if not SPOONACULAR_API_KEY:
        raise RuntimeError("SPOONACULAR_API_KEY not set in environment")
    ing_csv = ",".join([i.replace(" ", "+") for i in ingredients]) if ingredients else ""
    params = {"ingredients": ing_csv, "number": number, "ranking": 1, "ignorePantry": True, "apiKey": SPOONACULAR_API_KEY}
    if exclude:
        params["excludeIngredients"] = ",".join(exclude)
    url = f"{SPOON_BASE}/recipes/findByIngredients?{urlencode(params)}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def spoon_search_by_query(query, number=4):
    params = {"query": query, "number": number, "apiKey": SPOONACULAR_API_KEY}
    url = f"{SPOON_BASE}/recipes/complexSearch?{urlencode(params)}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])

def spoon_get_recipe_information(recipe_id):
    url = f"{SPOON_BASE}/recipes/{recipe_id}/information"
    params = {"apiKey": SPOONACULAR_API_KEY, "includeNutrition": False}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def format_recipe_for_chat(info: dict):
    title = info.get("title", "Recipe")
    ready = info.get("readyInMinutes")
    servings = info.get("servings")
    ingredients = [ing.get("originalString") for ing in info.get("extendedIngredients", [])]
    steps = []
    for sec in info.get("analyzedInstructions", []):
        for step in sec.get("steps", []):
            steps.append(step.get("step"))
    if not steps and info.get("instructions"):
        steps = [s.strip() for s in info["instructions"].split(". ") if s.strip()]
    parts = [f"**{title}**"]
    if ready:
        parts.append(f"⏱ Ready in: {ready} minutes")
    if servings:
        parts.append(f"🍽 Serves: {servings}")
    parts.append("**Ingredients:**")
    for ing in ingredients:
        parts.append(f"- {ing}")
    if steps:
        parts.append("**Steps:**")
        for i, s in enumerate(steps, 1):
            parts.append(f"{i}. {s}")
    if info.get("sourceUrl"):
        parts.append(f"🔗 Source: {info['sourceUrl']}")
    return "\n\n".join(parts)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Dobby Cooking Assistant", page_icon="🍳", layout="wide")
st.title("🍳 Dobby Cooking Assistant")

st.write("Type ingredients or ask for a dish. Example: 'I have potatoes and beef, what can I cook?'")

col1, col2 = st.columns([2, 1])
with col2:
    st.markdown("**Settings**")
    max_results = st.number_input("Results (max recipes):", min_value=1, max_value=8, value=3)
    show_images = st.checkbox("Show images", value=True)

# Chat storage
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []

# Chat container (messages)
chat_container = st.container()
with chat_container:
    for i in range(len(st.session_state.get("generated", []))):
        if i < len(st.session_state.get("past", [])):
            message(st.session_state["past"][i], is_user=True, key=f"user_{i}")
        message(st.session_state["generated"][i], key=f"gen_{i}")

# User input (always at bottom)
st.markdown("---")
user_text = st.text_input("Your question or ingredients:", key="input_text", placeholder="Ask Dobby...")
ask = st.button("Ask Dobby")

if ask and user_text:
    st.session_state.past.append(user_text)
    try:
        with st.spinner("Dobby is analyzing your request..."):
            parsed = analyze_user_text(user_text)
    except Exception as e:
        st.error(f"Analysis error: {e}")
        parsed = {"intent": "find_recipe", "ingredients": [user_text], "dish": None, "exclude": [], "message": f"Searching recipes for: {user_text}"}

    st.session_state.generated.append(f"Dobby: {parsed.get('message', 'I will search for recipes.')}")

    intent = parsed.get("intent")
    ingredients = parsed.get("ingredients", [])
    exclude = parsed.get("exclude", [])
    dish = parsed.get("dish")

    recipes_meta = []
    try:
        if intent in ("find_recipe", "modify_recipe"):
            if ingredients:
                results = spoon_search_by_ingredients(ingredients, exclude=exclude, number=max_results)
            else:
                q = dish or user_text
                results = spoon_search_by_query(q, number=max_results)
            for r in results:
                recipes_meta.append({"id": r.get("id"), "title": r.get("title"), "image": r.get("image")})
        elif intent == "specific_recipe":
            q = dish or user_text
            results = spoon_search_by_query(q, number=1)
            for r in results:
                recipes_meta.append({"id": r.get("id"), "title": r.get("title"), "image": r.get("image")})
        else:
            answer = call_fireworks_chat("You are Dobby, a helpful cooking assistant.", user_text, max_tokens=300, temperature=0.2)
            st.session_state.generated.append(answer)
    except Exception as e:
        st.error(f"Spoonacular search error: {e}")

    if recipes_meta:
        for meta in recipes_meta:
            try:
                info = spoon_get_recipe_information(meta["id"])
                formatted = format_recipe_for_chat(info)
                if show_images and meta.get("image"):
                    formatted = (
                        f'<img src="{meta["image"]}" width="800" '
                        f'style="border-radius:15px; box-shadow:0 4px 12px rgba(0,0,0,0.2); margin-bottom:15px;">\n\n'
                        + formatted
                    )
                st.session_state.generated.append(formatted)
            except Exception as e:
                st.session_state.generated.append(f"Could not fetch {meta.get('title')}: {e}")

# Scroll to bottom automatically
st.markdown(
    """
    <script>
        var chatDiv = window.parent.document.querySelector('.main');
        if (chatDiv) { chatDiv.scrollTo(0, chatDiv.scrollHeight); }
    </script>
    """,
    unsafe_allow_html=True,
)
