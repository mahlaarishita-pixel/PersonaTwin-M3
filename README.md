# PersonaTwin M3 — UK Financial Behaviour

## Run locally
1. Install Python 3.10–3.12.
2. Open a terminal in this folder.
3. Run:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Open the local URL printed by Streamlit.

## What is included
- Anonymous UK adult financial behaviour assessment
- Research-source annotations for each questionnaire item
- Six descriptive financial behaviour dimensions
- Three UK-pound financial scenarios
- CSV data collection and export
- Research and methodology page
- Descriptive dashboard
- Participant-level ML evaluation workflow (requires sufficient genuine responses)

## Research and ethics notes
- The OECD/INFE 2026 Toolkit is CC BY 4.0; attribution and disclosure of adaptations are required.
- MaPS Adult Financial Wellbeing Survey informs UK domain selection.
- Risk-tolerance prompts are project-written and informed by Grable & Lytton (1999); they are NOT the validated 13-item scale.
- Do not claim that a convenience sample represents all UK adults.
- Do not use synthetic records as evidence of real-world model accuracy.
- This is not financial advice, credit scoring or investment suitability assessment.
- Before public deployment, obtain supervisor/institutional approval, provide a complete privacy notice, set retention rules and use secure persistent storage.

## Research sources
- OECD (2026): https://doi.org/10.1787/92f2d439-en
- Money and Pensions Service (2022): https://maps.org.uk/en/publications/research/2021/financial-wellbeing-survey-2021
- Grable & Lytton (1999): https://doi.org/10.1016/S1057-0810(99)00041-4
- Kuzniak et al. (2015): https://doi.org/10.61190/fsr.v24i2.3240
