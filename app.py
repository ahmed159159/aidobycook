import os
import json
import requests
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from urllib.parse import urlencode
import base64

# ---------- Add Background Image ----------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# استدعاء الخلفية
add_bg_from_local("background.jpg")  # ← ضع الصورة في نفس مجلد المشروع

# ---------- Load .env ----------
load_dotenv()

# ---------- Config / env ----------
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "").strip()
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/ahmed159/deployedModels/dobby-unhinged-llama-3-3-70b-new-vdw6j81e")
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "").strip()

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
SPOON_BASE = "https://api.spoonacular.com"

# ---------- Helpers ----------
def call_fireworks_chat(system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.7):
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

# ---------- Analysis ----------
ANALYSIS_SYSTEM_PROMPT = """
You are Dobby, a friendly cooking assistant. Your job: analyze the user's message and return ONLY a compact JSON object (no other text) with the following fields:

- intent: one of "find_recipe", "specific_recipe", or "modify_recipe" or "general"
- ingredients: an array of ingredient words (lowercase) if present (or empty array)
- dish: a short dish name if user requested a specific recipe (or null)
- exclude: an array of ingredients to exclude (if user explicitly asked to avoid something)
- message: a short English sentence summarizing interpretation (for user display)

If the user asks something not about cooking, return {"intent":"general","message":"...","ingredients":[], "dish": null, "exclude": []}

Return JSON only, nothing else.
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
        return {"intent":"find_recipe", "ingredients":[user_text], "dish": None, "exclude": [], "message": f"Searching for recipes based on: {user_text}"}
    parsed.setdefault("ingredients", parsed.get("ingredients") or [])
    parsed.setdefault("exclude", parsed.get("exclude") or [])
    parsed.setdefault("dish", parsed.get("dish") if parsed.get("dish") is not None else None)
    parsed.setdefault("message", parsed.get("message") or "")
    parsed.setdefault("intent", parsed.get("intent") or "find_recipe")
    return parsed

# ---------- Spoonacular API ----------
def spoon_search_by_ingredients(ingredients, exclude=None, number=4):
    if not SPOONACULAR_API_KEY:
        raise RuntimeError("SPOONACULAR_API_KEY not set in environment")
    ing_csv = ",".join([i.replace(" ", "+") for i in ingredients]) if ingredients else ""
    params = {
        "ingredients": ing_csv,
        "number": number,
        "ranking": 1,
        "ignorePantry": True,
        "apiKey": SPOONACULAR_API_KEY
    }
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
    data = r.json()
    return data.get("results", [])

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
    ingredients = []
    for ing in info.get("extendedIngredients", []):
        amt = ing.get("originalString") or f"{ing.get('amount','')} {ing.get('unit','')} {ing.get('name','')}"
        ingredients.append(amt)
    steps = []
    ai = info.get("analyzedInstructions", [])
    if ai and isinstance(ai, list):
        for sec in ai:
            for step in sec.get("steps", []):
                steps.append(step.get("step"))
    if not steps:
        instr = info.get("instructions")
        if instr:
            steps = [s.strip() for s in instr.split(". ") if s.strip()]
    parts = []
    parts.append(f"**{title}**")
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
    source = info.get("sourceUrl")
    if source:
        parts.append(f"🔗 Source: {source}")
    return "\n\n".join(parts)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Dobby Cooking Assistant", page_icon="🍳", layout="wide")
st.title("Dobby Cooking Assistant — English (simple UI)")

st.write("Type ingredients or ask for a dish. Example: 'I have potatoes and 250g beef, what can I cook?'")
col1, col2 = st.columns([2,1])

with col2:
    st.markdown("**Settings**")
    max_results = st.number_input("Results (max recipes to fetch):", min_value=1, max_value=8, value=3)
    temp = st.slider("Dobby temperature (analysis)", min_value=0.0, max_value=1.0, value=0.2)
    show_images = st.checkbox("Show images (if available)", value=True)
    st.markdown("---")
    st.markdown("**Env status**")
    st.markdown(f"- Fireworks key: {'OK' if FIREWORKS_API_KEY else 'MISSING'}")
    st.markdown(f"- Spoonacular key: {'OK' if SPOONACULAR_API_KEY else 'MISSING'}")

# chat area
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []

user_text = st.text_input("Your question or ingredients:", key="input_text")
if st.button("Ask Dobby") and user_text:
    try:
        st.session_state.past.append(user_text)
        with st.spinner("Dobby is analyzing your request..."):
            parsed = analyze_user_text(user_text)
    except Exception as e:
        st.error(f"Analysis error: {e}")
        parsed = {"intent":"find_recipe", "ingredients":[user_text], "dish": None, "exclude": [], "message": f"Searching for recipes based on: {user_text}"}

    st.session_state.generated.append(f"Dobby: {parsed.get('message','I will search for recipes.')}")

    intent = parsed.get("intent")
    ingredients = parsed.get("ingredients") or []
    exclude = parsed.get("exclude") or []
    dish = parsed.get("dish") or None

    recipes_meta = []
    try:
        if intent in ("find_recipe","modify_recipe"):
            if ingredients:
                results = spoon_search_by_ingredients(ingredients, exclude=exclude, number=max_results)
                for r in results:
                    recipes_meta.append({"id": r.get("id"), "title": r.get("title"), "image": r.get("image")})
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
            try:
                answer = call_fireworks_chat("You are Dobby, a helpful cooking assistant.", user_text, max_tokens=300, temperature=temp)
                st.session_state.generated.append(answer)
            except Exception as e:
                st.error(f"Dobby error: {e}")
    except Exception as e:
        st.error(f"Spoonacular search error: {e}")

    if recipes_meta:
        for meta in recipes_meta:
            try:
                info = spoon_get_recipe_information(meta["id"])
                formatted = format_recipe_for_chat(info)
                if show_images and meta.get("image"):
                    formatted = f"![recipeimage]({meta.get('image')})\n\n" + formatted
                st.session_state.generated.append(formatted)
            except Exception as e:
                st.session_state.generated.append(f"Could not fetch details for {meta.get('title')}: {e}")

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=f"gen_{i}")
        if i < len(st.session_state.get("past", [])):
            message(st.session_state["past"][i], is_user=True, key=f"user_{i}")

st.markdown("---")
st.markdown("**Notes**: This app uses your Dobby model on Fireworks for analysis and Spoonacular for recipe data. Put keys in environment variables: FIREWORKS_API_KEY, FIREWORKS_MODEL (optional), SPOONACULAR_API_KEY.")
