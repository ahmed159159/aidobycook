import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from urllib.parse import urlencode
from pathlib import Path

# load .env locally
load_dotenv()

# ---------- Config / env ----------
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "").strip()
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/ahmed159/deployedModels/dobby-unhinged-llama-3-3-70b-new-vdw6j81e")
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "").strip()

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
SPOON_BASE = "https://api.spoonacular.com"

# assets paths
ASSETS_DIR = Path("assets")
USER_AVATAR = ASSETS_DIR / "user.png"
DOBBY_AVATAR = ASSETS_DIR / "dobby.png"
BACKGROUND = ASSETS_DIR / "background.jpg"

# ---------- Helpers (unchanged logic) ----------
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
    r = requests.post(FIREWORKS_URL, headers=headers, json=payload, timeout=120)
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

- intent: one of "find_recipe", "specific_recipe", or "modify_recipe" or "general"
- ingredients: an array of ingredient words (lowercase) if present (or empty array)
- dish: a short dish name if user requested a specific recipe (or null)
- exclude: an array of ingredients to exclude (if user explicitly asked to avoid something)
- message: a short English sentence summarizing interpretation (for user display)

If the user asks something not about cooking, return {"intent":"general","message":"...","ingredients":[], "dish": null, "exclude": []}

Return JSON only, nothing else.
"""

def analyze_user_text(user_text: str, temp: float = 0.1):
    raw = call_fireworks_chat(ANALYSIS_SYSTEM_PROMPT, user_text, max_tokens=256, temperature=temp)
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

# ---------- UI: Dark layout with left chat and right results ----------
st.set_page_config(page_title="Dobby Cooking Assistant", page_icon="🍳", layout="wide")

# hide Streamlit default menu/footer/header
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    html, body, [class*="css"]  { background-color: #0b0f12; }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# custom CSS for chat bubbles & layout
custom_css = """
<style>
.app-container {
  background-image: url('assets/background.jpg');
  background-size: cover;
  background-position: center;
  padding: 16px;
}
.container {
  background: rgba(10, 12, 15, 0.85);
  border-radius: 12px;
  padding: 16px;
  color: #e6edf3;
}
.left-column {
  padding: 12px;
  height: 75vh;
  overflow-y: auto;
  background: rgba(15,18,20,0.35);
  border-radius: 8px;
}
.right-column {
  padding: 12px;
  height: 75vh;
  overflow-y: auto;
  background: rgba(18,22,26,0.35);
  border-radius: 8px;
}
.msg-row {
  display: flex;
  align-items: flex-start;
  margin-bottom: 12px;
}
.msg-row.user { flex-direction: row-reverse; }
.msg-avatar {
  width: 42px;
  height: 42px;
  border-radius: 8px;
  margin: 0 8px;
  flex-shrink: 0;
  box-shadow: 0 2px 6px rgba(0,0,0,0.6);
}
.msg-bubble {
  max-width: 78%;
  padding: 10px 14px;
  border-radius: 12px;
  color: #e6edf3;
  line-height: 1.4;
}
.msg-bubble.user {
  background: linear-gradient(135deg,#1f6feb,#2cc2ff);
  border-bottom-right-radius: 4px;
}
.msg-bubble.dobby {
  background: linear-gradient(135deg,#1d1f24,#2b2f36);
  border-bottom-left-radius: 4px;
  color: #d6d9dd;
}
.recipe-card {
  background: rgba(8,10,12,0.7);
  border-radius: 10px;
  padding: 12px;
  margin-bottom: 12px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.5);
}
.recipe-title {
  font-size: 18px;
  font-weight: 600;
  color: #fff;
}
.recipe-meta {
  font-size: 13px;
  color: #c7d0da;
}
.recipe-image {
  width: 100%;
  border-radius: 8px;
  margin-bottom: 8px;
}
.input-row {
  display:flex;
  gap:8px;
  margin-top:8px;
}
.send-btn {
  background: linear-gradient(135deg,#ff7a18,#ff3d6b);
  color:white;
  padding:8px 12px;
  border-radius:8px;
  border:none;
}
.small-muted { color:#98a1ad; font-size:12px }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Page header
st.markdown("<div style='display:flex;align-items:center;gap:12px'><h2 style='margin:0;color:#fff'>Dobby Cooking Assistant</h2><div class='small-muted'>— powered by Dobby & Spoonacular</div></div>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>Type ingredients or ask for a dish. Example: 'I have potatoes and 250g beef, what can I cook?'</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# columns layout: left chat (2), right results (1)
col1, col2 = st.columns([2,1])

# Initialize session state
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []

# Input area (placed below chat, but we'll show controls here)
with col1:
    st.markdown("<div class='container'><div class='left-column' id='chat-area'>", unsafe_allow_html=True)

    # Render chat messages from session_state
    # messages are interleaved: past[] are user inputs; generated[] are Dobby replies and recipe blocks
    # We'll render user messages aligned right, Dobby short replies aligned left,
    # and treat generated items that look like recipes (contain 'Ingredients:' or start with '![') as results (do not show here).
    for idx, past_msg in enumerate(st.session_state.get("past", [])):
        # show user message bubble
        user_html = f"""
        <div class='msg-row user'>
          <img class='msg-avatar' src='assets/user.png' />
          <div class='msg-bubble user'>{st.session_state['past'][idx]}</div>
        </div>
        """
        st.markdown(user_html, unsafe_allow_html=True)

        # corresponding generated (if exists at same index or next)
        if idx < len(st.session_state.get("generated", [])):
            gen = st.session_state["generated"][idx]
            # if generated looks like a recipe (contains 'Ingredients:' or starts with image markdown), skip here
            if isinstance(gen, str) and ("**Ingredients:**" in gen or gen.strip().startswith("![" ) or gen.strip().startswith("**")):
                # skip showing recipe in chat (will be shown in right column)
                dobby_summary = ""  # optional: could show small acknowledgment
                # show a concise Dobby message above results (if you want)
                # but for now we show a short line (first line) as Dobby reply
                first_line = gen.splitlines()[0] if gen else ""
                if first_line:
                    dobby_html = f"""
                    <div class='msg-row dobby'>
                      <img class='msg-avatar' src='assets/dobby.png' />
                      <div class='msg-bubble dobby'>{first_line}</div>
                    </div>
                    """
                    st.markdown(dobby_html, unsafe_allow_html=True)
            else:
                # normal Dobby reply (show in chat)
                dobby_html = f"""
                <div class='msg-row dobby'>
                  <img class='msg-avatar' src='assets/dobby.png' />
                  <div class='msg-bubble dobby'>{gen}</div>
                </div>
                """
                st.markdown(dobby_html, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Input area (fixed under chat)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    with st.form(key="ask_form", clear_on_submit=False):
        user_input = st.text_input("Your question or ingredients:", key="input_text_form", placeholder="Ask me anything about food...", label_visibility="hidden")
        submit = st.form_submit_button("Send")
        if submit and user_input:
            # append to state and run analysis/search (same logic as before)
            try:
                st.session_state.past.append(user_input)
                # analysis
                parsed = analyze_user_text(user_input, temp=0.2)
            except Exception as e:
                st.error(f"Analysis error: {e}")
                parsed = {"intent":"find_recipe", "ingredients":[user_input], "dish": None, "exclude": [], "message": f"Searching for recipes based on: {user_input}"}

            # add Dobby short line to generated (displayed in chat as short)
            st.session_state.generated.append(parsed.get("message","I will search for recipes."))

            # now perform Spoonacular searches (same as before)
            intent = parsed.get("intent")
            ingredients = parsed.get("ingredients") or []
            exclude = parsed.get("exclude") or []
            dish = parsed.get("dish") or None

            recipes_meta = []
            try:
                if intent in ("find_recipe","modify_recipe"):
                    if ingredients:
                        results = spoon_search_by_ingredients(ingredients, exclude=exclude, number=3)
                        for r in results:
                            recipes_meta.append({"id": r.get("id"), "title": r.get("title"), "image": r.get("image")})
                    else:
                        q = dish or user_input
                        results = spoon_search_by_query(q, number=3)
                        for r in results:
                            recipes_meta.append({"id": r.get("id"), "title": r.get("title"), "image": r.get("image")})
                elif intent == "specific_recipe":
                    q = dish or user_input
                    results = spoon_search_by_query(q, number=1)
                    for r in results:
                        recipes_meta.append({"id": r.get("id"), "title": r.get("title"), "image": r.get("image")})
                else:
                    try:
                        answer = call_fireworks_chat("You are Dobby, a helpful cooking assistant.", user_input, max_tokens=300, temperature=0.4)
                        st.session_state.generated.append(answer)
                    except Exception as e:
                        st.error(f"Dobby error: {e}")
            except Exception as e:
                st.error(f"Spoonacular search error: {e}")

            # fetch and append full recipe details (these will be shown on right column)
            if recipes_meta:
                for meta in recipes_meta:
                    try:
                        info = spoon_get_recipe_information(meta["id"])
                        formatted = format_recipe_for_chat(info)
                        if meta.get("image"):
                            formatted = f"![recipeimage]({meta.get('image')})\n\n" + formatted
                        st.session_state.generated.append(formatted)
                    except Exception as e:
                        st.session_state.generated.append(f"Could not fetch details for {meta.get('title')}: {e}")

    # end left column
with col2:
    st.markdown("<div class='container'><div class='right-column'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#fff;margin-top:0'>🍽️ Results</h3>", unsafe_allow_html=True)

    # We will scan st.session_state["generated"] for recipe-like entries and render them as cards
    for gen in st.session_state.get("generated", []):
        if not isinstance(gen, str):
            continue
        # Identify recipe blocks (we inserted image markdown + **Title** and "**Ingredients:**" earlier)
        if ("**Ingredients:**" in gen) or gen.strip().startswith("![" ) or gen.strip().startswith("**"):
            # try to extract image url if present
            lines = gen.splitlines()
            img_url = None
            # if first line is markdown image
            if lines and lines[0].startswith("!["):
                # format: ![recipeimage](URL)
                start = lines[0].find("(")
                end = lines[0].find(")")
                if start != -1 and end != -1:
                    img_url = lines[0][start+1:end]
                    content_lines = lines[1:]
                else:
                    content_lines = lines
            else:
                content_lines = lines
            # title is first non-empty line (strip markdown ** if present)
            title = content_lines[0].strip() if content_lines else "Recipe"
            title = title.replace("**", "")
            # rest of content
            body = "\n".join(content_lines[1:]).strip() if len(content_lines) > 1 else ""
            # render card
            card_html = "<div class='recipe-card'>"
            if img_url:
                card_html += f"<img class='recipe-image' src='{img_url}'/>"
            card_html += f"<div class='recipe-title'>{title}</div>"
            if body:
                # render the body as markdown inside the card
                card_html += f"<div style='margin-top:8px;color:#cbd5e1'>{st.markdown(body, unsafe_allow_html=True)}</div>"
            card_html += "</div>"
            # write the html (note: st.markdown used above for body; to keep layout simple we also output html wrapper)
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

# Footer notes
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='small-muted' style='color:#9aa4b2'>Notes: Keep your keys in environment variables: FIREWORKS_API_KEY, FIREWORKS_MODEL (optional), SPOONACULAR_API_KEY</div>", unsafe_allow_html=True)
