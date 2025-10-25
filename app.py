import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from urllib.parse import urlencode

# ========== إعداد المفاتيح ==========
load_dotenv()
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/ahmed159/deployedModels/dobby-unhinged-llama-3-3-70b-new-vdw6j81e")
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "")

# ========== إعدادات API ==========
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
SPOON_BASE = "https://api.spoonacular.com"

# ========== دوال Fireworks ==========
def call_fireworks_chat(system_prompt, user_prompt, temperature=0.6):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIREWORKS_API_KEY}"
    }
    payload = {
        "model": FIREWORKS_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 600,
        "temperature": temperature
    }
    r = requests.post(FIREWORKS_URL, headers=headers, json=payload)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

# ========== تحليل سؤال المستخدم ==========
def analyze_user_text(user_text):
    system_prompt = """
You are Dobby, a helpful cooking assistant.
Analyze user's question and return a compact JSON with:
- intent: "find_recipe", "specific_recipe", or "modify_recipe"
- ingredients: array of ingredients (or empty)
- dish: dish name (if any)
- exclude: ingredients to exclude
- message: short friendly English line
Return JSON only.
"""
    try:
        result = call_fireworks_chat(system_prompt, user_text, temperature=0.2)
        cleaned = result.strip("```json").strip("```").strip()
        return json.loads(cleaned)
    except Exception:
        return {"intent":"find_recipe","ingredients":[user_text],"exclude":[],"dish":None,"message":"Let’s find something tasty!"}

# ========== Spoonacular ==========
def spoon_search_by_ingredients(ingredients, exclude=None, number=4):
    ing_csv = ",".join(ingredients)
    params = {
        "ingredients": ing_csv,
        "number": number,
        "ranking": 1,
        "apiKey": SPOONACULAR_API_KEY
    }
    if exclude:
        params["excludeIngredients"] = ",".join(exclude)
    r = requests.get(f"{SPOON_BASE}/recipes/findByIngredients?{urlencode(params)}")
    r.raise_for_status()
    return r.json()

def spoon_get_recipe_information(recipe_id):
    url = f"{SPOON_BASE}/recipes/{recipe_id}/information"
    params = {"apiKey": SPOONACULAR_API_KEY}
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

# ========== واجهة Streamlit ==========
st.set_page_config(page_title="🍳 Dobby Cooking Assistant", layout="wide")

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🍳 Dobby Cooking Assistant</h1>", unsafe_allow_html=True)

# تقسيم الشاشة
chat_col, results_col = st.columns([1.5, 1])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ========== قسم الشات ==========
with chat_col:
    st.markdown("<div class='chatbox'>", unsafe_allow_html=True)

    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"<div class='user-msg'>{chat['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{chat['content']}</div>", unsafe_allow_html=True)

    user_text = st.text_input("Type your message:", key="user_input", placeholder="e.g. I have chicken and rice... 🍗")
    send = st.button("Send")

    if send and user_text:
        st.session_state.chat_history.append({"role":"user", "content":user_text})
        parsed = analyze_user_text(user_text)
        dobby_reply = parsed.get("message", "Let's cook something amazing!")
        st.session_state.chat_history.append({"role":"bot", "content": f"Dobby: {dobby_reply}"})

        # البحث في Spoonacular
        ingredients = parsed.get("ingredients", [])
        recipes = spoon_search_by_ingredients(ingredients, number=4)

        st.session_state.recipes = recipes

    st.markdown("</div>", unsafe_allow_html=True)

# ========== قسم النتائج ==========
with results_col:
    st.markdown("<h3 class='section-title'>🍽️ Recipe Suggestions</h3>", unsafe_allow_html=True)

    recipes = st.session_state.get("recipes", [])
    if recipes:
        for r in recipes:
            st.markdown(f"""
            <div class='recipe-card'>
                <img src='{r["image"]}' class='recipe-img'>
                <div class='recipe-title'>{r["title"]}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recipes yet — ask Dobby about ingredients or a dish! 😋")
