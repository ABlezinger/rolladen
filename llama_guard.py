import streamlit as st
from openai import OpenAI

def check_safety_llama_guard_3(text):
    """
    Checks the safety of the given text using the LlamaGuard-3 model.
    """
    try:
        client = OpenAI(
            base_url=st.secrets["daisec_url"],
            api_key=st.secrets["DAISEC_API_KEY"]
        )

        response = client.chat.completions.create(
            model=st.secrets["safety_model"],
            messages=[{"role": "user", "content": text}],
            max_tokens=32000
        )

        safety_rating = response.choices[0].message.content
        print(f"safety_rating: {safety_rating}")

        if safety_rating == "safe":
            return True, None
        else:
            # Otherwise: Unsafe content detected, model outputs "Unsafe\n<Unsafe content categories>"
            # E.g., Unsafe\nS1,S2,S3
            categories = safety_rating.split("\n")[1].split(",")

            category_explanation = {
                "S1": "Gewaltverbrechen.",
                "S2": "Nicht-gewalttätige Verbrechen.",
                "S3": "Sexuelle Straftaten.",
                "S4": "Kinderausbeutung.",
                "S5": "Beleidigung und Verleumdung.",
                "S6": "Spezielle Beratung (z.B. medizinisch oder rechtlich).",
                "S7": "Datenschutz und Privatsphäre.",
                "S8": "Urheberrecht und geistiges Eigentum.",
                "S9": "Waffen und gefährliche Gegenstände.",
                "S10": "Hass und Diskriminierung.",
                "S11": "Selbstverletzung und Suizid.",
                "S12": "Sexuelle Inhalte.",
                "S13": "Wahlen und Politik.",
            }

            explanation = "Leider enthält deine Nachricht gewisse Inhalte, die als unsicher eingestuft wurden. Diese Inhalte sind:\n"
            for category in categories:
                explanation += f"- {category_explanation[category]}\n"
            explanation += "Bitte überprüfe deine Nachricht und versuche es erneut."

            return False, explanation
    except Exception as e:
        print(f"Error checking safety: {e}")
        return True, None