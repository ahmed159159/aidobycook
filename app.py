import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from urllib.parse import urlencode

# ----------- Load env -----------
load_dotenv()

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "").strip()
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/ahmed159/deployedModels/dobby-unhinged-llama-3-3-70b-new-vdw6j81e")
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "").strip()

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
SPOON_BASE = "https://api.spoonacular.com"

# ----------- Fireworks call -----------
def call_fireworks_chat(system_prompt, user_prompt, max_tokens=512, temperature=0.7):
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": FIREWORKS_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    r = requests.post(FIREWORKS_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()

# ----------- Analysis prompt -----------
ANALYSIS_SYSTEM_PROMPT = """
You are Dobby, a friendly cooking assistant. Your job: analyze the user's message and return ONLY JSON with:
- intent: "find_recipe", "specific_recipe", "modify_recipe" or "general"
- ingredients: list of words
- dish: short dish name or null
- exclude: ingredients to avoid
- message: short summary
Return JSON only, nothing else.
"""

def analyze_user_text(user_text):
    raw = call_fireworks_chat(ANALYSIS_SYSTEM_PROMPT, user_text, max_tokens=256, temperature=0.2)
    try:
        data = json.loads(raw.strip().split("```")[-1])
    except:
        data = {"intent": "find_recipe", "ingredients": [user_text], "dish": None, "exclude": [], "message": f"Looking for recipes with {user_text}"}
    return data

# ----------- Spoonacular ----------
def spoon_search_by_ingredients(ingredients, exclude=None, number=4):
    params = {
        "ingredients": ",".join(ingredients),
        "number": number,
        "ranking": 1,
        "ignorePantry": True,
        "apiKey": SPOONACULAR_API_KEY
    }
    if exclude:
        params["excludeIngredients"] = ",".join(exclude)
    url = f"{SPOON_BASE}/recipes/findByIngredients?{urlencode(params)}"
    r = requests.get(url, timeout=30)
    return r.json()

def spoon_get_recipe_information(recipe_id):
    url = f"{SPOON_BASE}/recipes/{recipe_id}/information"
    params = {"apiKey": SPOONACULAR_API_KEY}
    r = requests.get(url, params=params, timeout=30)
    return r.json()

# ----------- Format recipe ----------
def format_recipe(info):
    title = info.get("title", "Recipe")
    img = info.get("image")
    ready = info.get("readyInMinutes", "?")
    servings = info.get("servings", "?")
    steps = []
    for sec in info.get("analyzedInstructions", []):
        for step in sec.get("steps", []):
            steps.append(step["step"])
    html = f"""
    <div style='background:#222;padding:20px;border-radius:15px;margin-bottom:15px;color:#eee'>
        <h3 style='color:#FFD700'>{title}</h3>
        {'<img src="'+img+'" width="100%" style="border-radius:10px;margin-bottom:10px;">' if img else ''}
        <p>⏱ Ready in: {ready} min | 🍽 Serves: {servings}</p>
        <ol>{"".join(f"<li>{s}</li>" for s in steps[:10])}</ol>
    </div>
    """
    return html

# ----------- UI setup ----------
st.set_page_config(page_title="Dobby Cook Assistant", layout="wide")

# background style
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: url('assets/background.jpg');
    background-size: cover;
    background-attachment: fixed;
}
.chat-box {
    background: rgba(0,0,0,0.6);
    padding: 20px;
    border-radius: 15px;
    color: #eee;
}
.user-msg, .bot-msg {
    display: flex;
    align-items: flex-start;
    margin-bottom: 10px;
}
.user-msg img, .bot-msg img {
    width: 40px; height: 40px; border-radius: 50%;
    margin-right: 10px;
}
.bot-msg p, .user-msg p {
    background: #333;
    padding: 10px 15px;
    border-radius: 10px;
    color: #eee;
}
.user-msg p { background: #0078FF; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🍳 Dobby Cooking Assistant")

col_chat, col_results = st.columns([1.3, 1.2])

if "history" not in st.session_state:
    st.session_state["history"] = []
if "recipes" not in st.session_state:
    st.session_state["recipes"] = []

with col_chat:
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    for sender, msg in st.session_state["history"]:
        if sender == "user":
            st.markdown(f"<div class='user-msg'><img src='assets/user.png'><p>{msg}</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'><img src='assets/dobby.png'><p>{msg}</p></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    user_text = st.text_input("Type your message:", key="input")
    if st.button("Send"):
        if user_text.strip():
            st.session_state["history"].append(("user", user_text))
            with st.spinner("Dobby is thinking..."):
                parsed = analyze_user_text(user_text)
                msg = parsed.get("message", "Let’s see what I can find for you.")
                st.session_state["history"].append(("bot", msg))
                if parsed["intent"] == "find_recipe":
                    try:
                        results = spoon_search_by_ingredients(parsed["ingredients"])
                        st.session_state["recipes"] = [spoon_get_recipe_information(r["id"]) for r in results]
                    except Exception as e:
                        st.session_state["history"].append(("bot", f"Error: {e}"))


with col_results:
    st.markdown("## 🍽 Recipes")
    if not st.session_state["recipes"]:
        st.info("No recipes yet. Ask Dobby what to cook!")
    else:
        for r in st.session_state["recipes"]:
            st.markdown(format_recipe(r), unsafe_allow_html=True)
