import streamlit as st
from openai import OpenAI
from system_prompts import system_prompt
from fpdf import FPDF
from datetime import datetime
import re

def _get_stream_content(chunk):
    """Safely extract content string from a streaming chunk, or None if unavailable."""
    try:
        choices = getattr(chunk, "choices", None)
        if not choices:
            return None
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            return None
        return getattr(delta, "content", None)
    except Exception:
        return None

def convert_latex_math_to_unicode(line):
    replacements = [
        (r'\\Omega', 'Ω'),
        (r'\\cdot', '·'),
        (r'\\mu', 'µ'),
        (r'\\rho', 'ρ'),
        (r'\^2', '²'),
        (r'\^3', '³'),
        (r'_([0-9])', lambda m: chr(0x2080 + int(m.group(1)))),
        (r'\\,', ' '),
    ]
    line = line.replace('$', '')
    for pattern, repl in replacements:
        line = re.sub(pattern, repl, line)
    return line

def preprocess_pdf_line(line):
    # Entferne **...** (Markdown fett)
    line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)

    # Latex-Befehle: \text{ ... } zu Inhalt ohne Latex
    line = re.sub(r"\\text\s*\{\s*([^}]*)\s*\}", r"\1", line)

    # Hoch- und Tiefstellungen: _{ges} und ^{2}
    line = re.sub(r"_\{([a-zA-Z0-9]+)\}", lambda m: to_subscript(m.group(1)), line)
    line = re.sub(r"\^\{([a-zA-Z0-9]+)\}", lambda m: to_superscript(m.group(1)), line)

    # Hochzahlen wie ^2, ^3 (ohne Klammern)
    line = re.sub(r"\^2", "²", line)
    line = re.sub(r"\^3", "³", line)

    # Greek: \Omega, \mu, \rho
    line = line.replace(r'\Omega', 'Ω')
    line = line.replace(r'\mu', 'µ')
    line = line.replace(r'\rho', 'ρ')
    line = line.replace(r'\cdot', '·')

    # Restliche $ entfernen
    line = line.replace('$', '')
    # überflüssige Latex-Trenner
    line = line.replace(r'\,', ' ')
    return line

def to_subscript(s):
    # Nur (lateinische) Buchstaben und Ziffern zu Unicode-Subscript
    normal = "0123456789aehijklmnoprstuvx+-=()"
    sub =    "₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ₊₋₌₍₎"
    trans = str.maketrans(normal, sub)
    return ''.join([c.translate(trans) if c in trans else c for c in s])

def to_superscript(s):
    # Nur Ziffern und Plus/Minus zu Superscript
    trans = str.maketrans('0123456789+-=()', '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾')
    return ''.join([c.translate(trans) if c in trans else c for c in s])

def wrap_text_by_chars(text, max_len=75):
    """Wrap text at a maximum number of characters per line.

    Breaks at the last whitespace at or before max_len when possible,
    otherwise performs a hard break at max_len. Whitespace counts toward the limit.
    """
    lines = []
    remaining = text
    while len(remaining) > max_len:
        break_pos = remaining.rfind(' ', 0, max_len + 1)
        if break_pos == -1 or break_pos == 0:
            break_pos = max_len
        lines.append(remaining[:break_pos].rstrip())
        remaining = remaining[break_pos:].lstrip()
    if remaining:
        lines.append(remaining)
    return lines

def run_pdf_generator(aufgaben_content, client):
    """
    Generate a formatted PDF exam file from plain text tasks.

    Parameters
    ----------
    tasks_content : str
        Raw exam text containing tasks without solutions.
    client : OpenAI
        OpenAI client for generating formatted exam content.

    Returns
    -------
    bytes
        PDF file as byte stream for download.
    """
    st.divider()

    # --- Response State ---
    pdf_response = ""
    thinking_text = ""
    in_thinking_block = False

    # --- Generation Phase: Request PDF-ready content from LLM ---
    with st.status("📄 PDF wird generiert...", expanded=True) as status:
        # Systemprompt erstellen
        syst_prompt = system_prompt(role="bbs_pdf_generator",)

        # API request for converting the raw exam text into a clean document format
        response = client.chat.completions.create(
            model="gemma-3-27b-it",#"gemma3:27b-it-q8_0",
            messages=[
                {"role": "system", "content": syst_prompt},
                {"role": "user", "content": f"""
                    Formatiere den folgenden Klausurtext für FPDF.
                    Gib ausschließlich den reinen, formatierten Klausurtext zurück – ohne Einleitung, ohne Erklärung, ohne Anmerkungen, ohne Zwischenschritte.
                    Entferne alle Lösungen.
                    Gib nur den Text zurück, der direkt und ohne weitere Bearbeitung in die PDF eingefügt werden kann.
                    Alles andere als das genannte ist ein Fehler.
                    Textinhalt:
                    {aufgaben_content}
                    
                    Beachte:
                    1. Verwende ## für Hauptüberschriften (zentriert, fett)
                    2. Verwende ### für Unterüberschriften (linksbündig, fett)
                    3. Aufgaben beginnen mit **Aufgabe X** (fett, unterstrichen)
                    4. Punkte stehen in Klammern nach jeder Frage
                    5. Lösungen vollständig entfernen
                    6. Füge --- nach jeder Aufgabe als Trennlinie ein
                    7. Gesamtpunktzahl am Ende deutlich angeben
                    8. Konsistente Abstände: 5mm nach Überschriften, 3mm nach Aufgaben
                    """
                }
            ],
            stream=True,
        )

        # Collect streaming response, skipping hidden "thinking" sections
        for chunk in response:
            content = _get_stream_content(chunk)
            if content is None or content == "":
                continue

            if "<think>" in content:
                in_thinking_block = True
                content = content.replace("<think>", "")
            if "</think>" in content:
                in_thinking_block = False
                thinking_text += content.replace("</think>", "")
                continue

            if in_thinking_block:
                thinking_text += content
                continue 

            # Append actual exam text
            pdf_response += content
        status.update(label="✅ PDF generiert", state="complete")

    # --- Formatting Phase: Convert text into PDF structure ---
    with st.status("📄 PDF wird formatiert...", expanded=True) as status:
        pdf = FPDF()
        pdf.add_page()

        # Register custom fonts
        pdf.add_font('ArialUnicode', '', 'Arial-Unicode-MS.ttf', uni=True)
        pdf.add_font('ArialUnicode', 'B', 'Arial-Unicode-MS.ttf', uni=True)
        pdf.add_font('ArialUnicode', 'I', 'Arial-Unicode-MS.ttf', uni=True)

        # Layout and margins
        pdf.set_margins(25, 15, 25)  # Links, Oben, Rechts
        pdf.set_auto_page_break(True, margin=15)
        
        # Parse and render lines
        for line in pdf_response.split('\n'):
            line = convert_latex_math_to_unicode(line.strip())
            line = preprocess_pdf_line(line.strip())
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("## "):
                # Main heading
                pdf.set_font('ArialUnicode', 'B', 16)
                pdf.cell(0, 10, line[3:], ln=True, align='C')
                pdf.ln(5)
                
            elif line.startswith("### "):
                # Subheading
                pdf.set_font('ArialUnicode', 'B', 14)
                pdf.cell(0, 8, line[4:], ln=True)
                pdf.ln(3)
                
            elif line.startswith("**Aufgabe"):
                # Task heading
                pdf.set_font('ArialUnicode', 'B', 12)
                pdf.set_text_color(0, 0, 128)  # Dunkelblau
                pdf.cell(0, 7, line[2:-2], ln=True)  # Entfernt **
                pdf.set_text_color(0, 0, 0)  # Schwarz zurück
                pdf.ln(2)

            elif line.startswith("---"):
                # Separator line
                pdf.ln(5)
                pdf.line(25, pdf.get_y(), 180, pdf.get_y())
                pdf.ln(8)
                
            elif "Punkte:" in line or "(" in line:
                # Points annotation
                pdf.set_font('ArialUnicode', 'I', 12)
                for seg in wrap_text_by_chars(line, 75):
                    pdf.cell(0, 7, seg, ln=True)
                
            else:
                # Regular body text
                pdf.set_font('ArialUnicode', '', 12)
                for seg in wrap_text_by_chars(line, 75):
                    pdf.cell(0, 6, seg, ln=True)
                

        # Final total points section
        pdf.ln(10)
        pdf.set_font('ArialUnicode', 'B', 14)

        status.update(label="✅ PDF formatiert", state="complete")

    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return pdf_bytes

@st.fragment
def run_pdf_download(pdf_bytes, selected_fach):
    """
    Provide a download button for the generated PDF.

    Parameters
    ----------
    pdf_bytes : bytes
        Content of the PDF file as byte stream.
    selected_subject : str
        Subject name to include in the filename.
    """
    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name=f"Kompetenztest_{selected_fach}_{datetime.now().strftime('%Y-%m-%d')}.pdf",
        mime="application/pdf"
    )