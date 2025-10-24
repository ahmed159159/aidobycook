import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message

# ---------- Load Environment ----------
load_dotenv()
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/ahmed159/deployedModels/dobby-unhinged-llama-3-3-70b-new-vdw6j81e")
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "")

# ---------- Helper Functions ----------
def analyze_user_text(user_text):
    return {"intent": "find_recipe", "ingredients": [user_text], "dish": None, "exclude": [], "message": "Based on your ingredients, you can try these dishes:"}

def spoon_search_by_ingredients(ingredients, number=3):
    url = f"https://api.spoonacular.com/recipes/findByIngredients?ingredients={','.join(ingredients)}&number={number}&apiKey={SPOONACULAR_API_KEY}"
    r = requests.get(url)
    return r.json()

def spoon_get_recipe_information(recipe_id):
    url = f"https://api.spoonacular.com/recipes/{recipe_id}/information?apiKey={SPOONACULAR_API_KEY}"
    r = requests.get(url)
    return r.json()

def format_recipe(info):
    title = info.get("title", "Recipe")
    img = info.get("image", "")
    steps = []
    for section in info.get("analyzedInstructions", []):
        for step in section.get("steps", []):
            steps.append(step.get("step"))
    steps_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)]) or "No steps found."
    return f"**{title}**\n\n![img]({img})\n\n{steps_text}"

# ---------- Page Layout ----------
st.set_page_config(page_title="🍳 Dobby Cooking Assistant", layout="wide")

# Apply CSS style
with open("assets/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# ---------- Columns ----------
col_chat, col_result = st.columns([2, 1])

# Session State
if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "results" not in st.session_state:
    st.session_state["results"] = []

# ---------- Chat Column ----------
with col_chat:
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    st.title("💬 Chat with Dobby")

    # Chat area (scroll)
    chat_container = st.container(height=500)
    with chat_container:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=f"user_{i}")
            message(st.session_state["generated"][i], key=f"bot_{i}")

    # Input box + Send button inside chat
    with st.container():
        user_input = st.text_input("Type your question or ingredients...", key="input", placeholder="e.g. I have potatoes and beef")
        send_btn = st.button("Send", key="send")

        if send_btn and user_input:
            st.session_state["past"].append(user_input)
            parsed = analyze_user_text(user_input)
            st.session_state["generated"].append(f"Dobby: {parsed['message']}")

            # Fetch recipes safely
            try:
                data = spoon_search_by_ingredients(parsed["ingredients"])
                if isinstance(data, list):
                    st.session_state["results"] = [r for r in data if isinstance(r, dict)]
                else:
                    st.session_state["results"] = []
            except Exception as e:
                st.session_state["generated"].append(f"⚠️ Error fetching recipes: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Results Column ----------
with col_result:
    st.markdown("<div class='results-box'>", unsafe_allow_html=True)
    st.title("🍽️ Results")
    if st.session_state["results"]:
        for recipe in st.session_state["results"]:
            rid = recipe.get("id")
            title = recipe.get("title", "")
            img = recipe.get("image", "")
            st.markdown(f"**{title}**")
            if img:
                st.image(img, use_column_width=True)
            try:
                info = spoon_get_recipe_information(rid)
                st.markdown(format_recipe(info))
            except Exception:
                st.markdown("⚠️ Could not load full recipe info.")
            st.markdown("---")
    st.markdown("</div>", unsafe_allow_html=True)
