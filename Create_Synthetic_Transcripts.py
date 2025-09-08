import os
import time
import json
import logging
import sys
import random
import glob
from datetime import datetime
from functools import wraps
from mistralai import Mistral
from openai import OpenAI

RATE_LIMIT_DELAY = float(os.environ.get("RATE_LIMIT_DELAY", 3.0))
DEFAULT_NUM_TURNS = int(os.environ.get("NUM_TURNS", 25))
DEFAULT_SLIDING_WINDOW_TOKENS = int(os.environ.get("SLIDING_WINDOW_TOKENS", 30000))
RETRY_MAX_ATTEMPTS = int(os.environ.get("RETRY_MAX_ATTEMPTS", 10))
RETRY_BASE_DELAY = float(os.environ.get("RETRY_BASE_DELAY", 3.0))
SUPERVISION_ENABLED = os.environ.get("SUPERVISION_ENABLED", "true").lower() in ("true", "1", "yes")
PATIENT_VALIDATION_ENABLED = os.environ.get("PATIENT_VALIDATION_ENABLED", "true").lower() in ("true", "1", "yes")
MAX_REVISION_ATTEMPTS = int(os.environ.get("MAX_REVISION_ATTEMPTS", 2))
NUM_SYNTHETIC_PATIENTS = int(os.environ.get("NUM_SYNTHETIC_PATIENTS", 50))

def generate_synthetic_patient_profile():
    """Generates a more nuanced synthetic patient profile."""

    age_options = [
        (18, 24, "a young adult"), (25, 34, "an adult in their late twenties or early thirties"),
        (35, 49, "a middle-aged individual"), (50, 64, "an individual in their early fifties to mid-sixties"),
        (65, 79, "a senior individual"), (80, 99, "an elderly individual"),
    ]
    age = random.choice(age_options)
    age_range, age_description = f"{age[0]}-{age[1]}", age[2]
    gender = random.choice(["male", "female", "non-binary"])
    occupations = [
        "software developer", "teacher", "nurse", "artist", "accountant", "student", "manager",
        "construction worker", "chef", "social worker", "business owner", "unemployed", "data scientist"
    ]
    occupation = random.choice(occupations)
    mental_health_issues = [
        ("mild anxiety", "occasional panic attacks, general worry"),
        ("moderate depression", "low energy, difficulty concentrating, loss of interest"),
        ("generalized anxiety disorder", "persistent worry, restlessness, muscle tension"),
        ("social anxiety", "intense fear of social judgment, avoidance of social situations"),
        ("PTSD", "flashbacks, nightmares, hypervigilance related to past trauma"),
        ("OCD", "intrusive thoughts, compulsive behaviors (e.g., checking, cleaning)"),
        ("burnout", "emotional exhaustion, cynicism, reduced efficacy related to work/stress"),
        ("adjustment disorder", "difficulty coping with a specific stressor (e.g., move, job change)"),
        ("low self-esteem", "pervasive feelings of inadequacy, harsh self-criticism"),
        ("grief", "prolonged sadness, difficulty functioning after a significant loss")
    ]
    mental_health_issue, symptom_description = random.choice(mental_health_issues)
    life_events = [
        "a recent difficult breakup", "the loss of a loved one", "job loss or instability",
        "a recent move", "ongoing financial stress", "starting a demanding new job or school program",
        "significant family conflict", "a health scare", "feeling isolated or lonely", "major life transition (e.g., empty nest)"
    ]
    life_event = random.choice(life_events)
    personalities = [
        ("introverted", "analytical", "cautious"), ("extroverted", "expressive", "action-oriented"),
        ("reserved", "detail-oriented", "anxious"), ("outgoing", "adaptable", "sometimes impulsive"),
        ("calm", "thoughtful", "private"), ("sensitive", "creative", "prone to self-doubt"),
        ("pragmatic", "organized", "skeptical"), ("gregarious", "optimistic", "easily distracted")
    ]
    personality1, personality2, personality3 = random.choice(personalities)
    coping_mechanisms = [
        "talking to friends/family", "avoiding triggers", "engaging in hobbies", "exercise",
        "mindfulness/meditation", "overworking", "substance use (mild/moderate)", "seeking reassurance",
        "intellectualizing feelings", "emotional eating", "procrastination", "using humor/sarcasm"
    ]
    coping_mechanism = random.choice(coping_mechanisms)
    backgrounds = [
        "Grew up in a stable but emotionally reserved family.",
        "Had a somewhat chaotic childhood with inconsistent parenting.",
        "Comes from a high-achieving family, feels pressure to succeed.",
        "Experienced bullying in school, affecting social confidence.",
        "Has a history of difficult romantic relationships.",
        "Recently moved away from their primary support system.",
        "Struggled academically but found success later in their career.",
        "Has always been independent, sometimes finding it hard to ask for help."
    ]
    background = random.choice(backgrounds)
    relationship_statuses = ["single", "in a relationship", "married", "divorced", "widowed"]
    relationship_status = random.choice(relationship_statuses)
    support_systems = [
        "a few close friends", "a supportive partner", "limited social support currently",
        "supportive family (nearby or distant)", "relies mostly on self", "colleagues provide some support"
    ]
    support_system = random.choice(support_systems)

    interaction_styles_and_mindedness = [
        ("Guarded/Hesitant", "Low", "Initially reserved, takes time to open up, may be skeptical of therapy exercises."),
        ("Intellectualizing", "Moderate", "Analyzes feelings rather than experiencing them, uses abstract language, may resist experiential exercises."),
        ("Eager/Compliant", "Moderate", "Wants to 'do therapy right', agrees readily but may struggle to apply concepts outside session."),
        ("Overwhelmed/Distractible", "Low", "Easily sidetracked, finds it hard to focus, may need concepts broken down simply."),
        ("Pragmatic/Concrete", "Low", "Prefers practical solutions, impatient with abstract ideas, asks 'what's the point?'."),
        ("Emotionally Expressive", "Moderate", "Openly shares feelings but may struggle to connect them to values or actions."),
        ("Curious/Engaged", "High", "Willing to try exercises, asks clarifying questions, grasps concepts relatively quickly but still faces struggles."),
        ("Argumentative/Resistant", "Low", "Challenges therapist's suggestions, expresses strong doubts, may focus on perceived flaws in ACT."),
        ("Quiet/Reflective", "High", "Speaks thoughtfully, internalizes concepts but may need encouragement to verbalize insights.")
    ]
    interaction_style, psych_mindedness, style_description = random.choice(interaction_styles_and_mindedness)

    presenting_problems_detail = [
        f"Struggling with constant worry about performance at their job as a {occupation}, leading to procrastination.",
        f"Feeling overwhelmed by sadness and lack of motivation since {life_event}, impacting their relationship.",
        f"Experiencing intense anxiety in social settings, causing them to avoid gatherings with friends ({support_system}).",
        f"Caught in cycles of harsh self-criticism related to perceived failures, linked to {background.lower()}",
        f"Difficulty managing anger and frustration, especially in interactions related to {life_event}.",
        f"Feeling stuck and directionless, unsure what matters to them beyond ({occupation}).",
        f"Using {coping_mechanism} to numb uncomfortable feelings related to {mental_health_issue}."
    ]
    presenting_problem = random.choice(presenting_problems_detail)

    profile = (
        f"Patient is {age_description} ({age_range}), identifies as {gender}, works as a {occupation}, and is currently {relationship_status}. "
        f"Primary concern involves {mental_health_issue} ({symptom_description}), particularly manifesting as: {presenting_problem}. "
        f"This seems exacerbated by {life_event}. {background} Their typical coping mechanism is {coping_mechanism}. "
        f"Personality traits include being {personality1}, {personality2}, and {personality3}. They have {support_system}. "
        f"Interaction Style: {interaction_style} ({style_description}). Psychological Mindedness: {psych_mindedness}."
    )

    return profile, presenting_problem, interaction_style, psych_mindedness

log_filename = None

mistral_api_key = os.environ.get("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not set.")
mistral_client = Mistral(api_key=mistral_api_key)
generator_model_name = "mistral-large-latest"

deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set for supervisor/validator.")
deepseek_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
supervisor_model_name = "deepseek-chat"

def retry_with_backoff(max_retries=RETRY_MAX_ATTEMPTS, base_delay=RETRY_BASE_DELAY, allowed_exceptions=(Exception,)):
    """ Decorator for retrying a function with exponential backoff. """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    attempts += 1
                    if attempts >= max_retries:
                        logging.error(f"Function {func.__name__} failed after {max_retries} attempts: {str(e)}")
                        raise
                    wait_time = base_delay * (2 ** min(attempts - 1, 4)) + random.uniform(0, 1)
                    logging.warning(f"Attempt {attempts}/{max_retries} failed for {func.__name__}. Retrying in {wait_time:.2f}s. Error: {str(e)}")
                    time.sleep(wait_time)
            raise Exception(f"Max retries exceeded for function {func.__name__}")
        return wrapper
    return decorator

class RateLimiter:
    """ Simple rate limiter to ensure minimum delay between calls. """
    def __init__(self, min_delay=RATE_LIMIT_DELAY):
        self.min_delay = min_delay
        self.last_request_time = 0

    def wait(self):
        """ Waits if necessary to maintain the minimum delay. """
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_delay:
            sleep_time = self.min_delay - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()

shared_rate_limiter = RateLimiter(min_delay=RATE_LIMIT_DELAY)

def validate_patient_response_format(response):
    """Validates basic formatting for a patient response."""
    if not isinstance(response, str) or not response or response.isspace():
        raise ValueError("Patient response must be a non-empty string.")
    forbidden_patterns = [
        "Background:", "Patient Scenario:", "Your Task:", "Your Response:",
        "Transcript:", "<|thinking|>", "<|answer|>", "Therapist:", "Patient:",
        "---",
        "Output Format:", "Evaluation Context:", "VERDICT", "<|feedback|>"
    ]
    response_lower = response.lower()
    if any(pattern.lower() in response_lower for pattern in forbidden_patterns):
        raise ValueError(f"Patient response contains prompt artifacts or speaker tags. Found patterns like: {[p for p in forbidden_patterns if p.lower() in response_lower]}")
    if "**" in response or "`" in response:
        raise ValueError("Patient response contains markdown formatting (e.g., **, * `).")
    if response.strip().endswith(("-", "—")) and not response.strip().endswith("..."):
         raise ValueError("Patient response appears truncated (ends with dash/em-dash).")
    return response.strip()

def validate_therapist_response(thinking, answer):
    """Validates basic formatting for a therapist response."""
    if not isinstance(thinking, str) or not thinking or thinking.isspace():
        if thinking != "[THINKING TAG MISSING]":
             raise ValueError("Thinking part must be a non-empty string.")
    if not isinstance(answer, str) or not answer or answer.isspace():
        if answer != "[ANSWER TAG MISSING OR MISPLACED]":
            raise ValueError("Answer part must be a non-empty string.")
    forbidden_in_answer = ["Background:", "Patient Scenario:", "Patient:", "Your Response:", "---"]
    answer_lower = answer.lower()
    if any(pattern.lower() in answer_lower for pattern in forbidden_in_answer):
        raise ValueError(f"Therapist answer contains background info, patient tags, or prompt remnants. Found patterns like: {[p for p in forbidden_in_answer if p.lower() in answer_lower]}")

    if answer.strip().endswith(("-", "—")) and not answer.strip().endswith("..."):
        raise ValueError("Therapist answer appears truncated (ends with dash/em-dash).")
    return thinking.strip(), answer.strip()

def validate_conversation_coherence(transcript, new_entry, role):
    """Checks if the new entry is identical to the last entry by the same role."""
    assert isinstance(transcript, list), "Transcript must be a list."
    assert isinstance(new_entry, str), "New entry must be a string."
    assert role in ["Patient", "Therapist"], "Role must be 'Patient' or 'Therapist'."

    if transcript:
        last_relevant_entry = None
        for message in reversed(transcript):
             if message.startswith(f"{role}:"):
                last_relevant_entry = message
                break

        if last_relevant_entry:
            last_content = last_relevant_entry[len(f"{role}:"):].strip()
            if role == "Therapist" and "<|answer|>" in last_content:
                 last_content = last_content.split("<|answer|>", 1)[-1].strip()

            new_content = new_entry[len(f"{role}:"):].strip()
            if role == "Therapist" and "<|answer|>" in new_content:
                new_content = new_content.split("<|answer|>", 1)[-1].strip()

            if len(new_content) > 25 and new_content == last_content:
                raise ValueError(f"New {role} response content is identical to the previous one, potential loop. Content: '{new_content[:50]}...'")

    return new_entry

def save_conversation_state(state, file_path):
    """Saves the current conversation state to a JSON file."""
    assert isinstance(state, dict), "State must be a dictionary."
    assert isinstance(file_path, str), "File path must be a string."
    try:
        temp_file_path = file_path + ".tmp"
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(temp_file_path, file_path)
        logging.info(f"Conversation state saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving conversation state to {file_path}: {e}")
        if os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except OSError: pass
        raise

def load_conversation_state(file_path):
    """Loads the conversation state from a JSON file."""
    assert isinstance(file_path, str), "File path must be a string."
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"State file not found: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        logging.info(f"Conversation state loaded from {file_path}")
        assert isinstance(state, dict), "Loaded state is not a dictionary."
        required_keys = ["full_transcript", "clean_transcript", "training_pairs",
                         "turn_number", "last_therapist_answer", "patient_scenario",
                         "presenting_problem", "patient_interaction_style", "patient_psych_mindedness"]
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
             raise ValueError(f"Loaded state is missing required key(s): {', '.join(missing_keys)}")

        assert isinstance(state.get("full_transcript"), list), "Loaded 'full_transcript' is not a list."
        assert isinstance(state.get("clean_transcript"), list), "Loaded 'clean_transcript' is not a list."
        assert isinstance(state.get("training_pairs"), list), "Loaded 'training_pairs' is not a list."
        assert isinstance(state.get("turn_number"), int), "Loaded 'turn_number' is not an integer."
        assert isinstance(state.get("last_therapist_answer"), str), "Loaded 'last_therapist_answer' is not a string."
        expected_len_at_turn_n = 1 + 2 * state["turn_number"]
        if len(state.get("clean_transcript", [])) != expected_len_at_turn_n:
             logging.warning(f"Loaded state clean_transcript length ({len(state.get('clean_transcript', []))}) "
                             f"does not match expected length for turn {state['turn_number']} ({expected_len_at_turn_n}). "
                             f"May indicate interruption after therapist turn before state save. Resuming normally.")

        return state
    except (json.JSONDecodeError, AssertionError, ValueError, FileNotFoundError) as e:
        logging.error(f"Error loading or validating state from {file_path}: {e}")
        raise

therapist_system_prompt = """You are an AI simulating an Acceptance and Commitment Therapy (ACT) therapist. Your goal is to generate realistic, varied training data. Adhere strictly to ACT principles and maintain a natural, flexible style.

**Core Directives:**
1.  **Elicit Context FIRST:** Before introducing core ACT exercises/metaphors, actively elicit details about the patient's presenting problem, psychosocial context (work, relationships, recent events), and how the problem manifests in their daily life. Ask clarifying questions. Refer back to these elicited details throughout the session.
2.  **Realistic Pacing & Flexibility:** Introduce concepts GRADUALLY. Avoid rushing through ACT processes. If a patient is confused, resistant, or struggling with an exercise (based on their Interaction Style or response), ACKNOWLEDGE this and adapt. Slow down, simplify, revisit earlier themes, or gently explore the resistance itself using ACT principles. Progress may be non-linear.
3.  **Focused Communication:** Each therapist turn (<|answer|>) should have ONE primary focus – either a single core question, a single reflection, or a single brief instruction. AVOID asking multiple distinct questions or requesting multiple actions simultaneously.
4.  **Varied & Natural Language:** CONSCIOUSLY VARY phrasing. Avoid overusing stock phrases like "It sounds like...", "Let's explore...", "Take a moment...", "How does that land?", "What comes up for you?". Use synonyms, different sentence structures, and diverse validation techniques. Sound like a human therapist, not a script.
5.  **Conciseness:** Keep both <|thinking|> and <|answer|> concise. Aim for <|thinking|> around 50-150 words, clearly justifying the *immediate* intervention choice based on the patient's last statement and the overall flow. Aim for <|answer|> to be natural dialogue length (usually 1-4 sentences), unless a longer explanation is therapeutically necessary and justified. Reduce fluff.
6.  **Justified Interventions:** Use a range of ACT processes (Defusion, Acceptance, Present Moment, Values, Committed Action, Self-as-Context) but justify their use in <|thinking|> based on the *current* conversational turn and patient state. Use metaphors *sparingly* and only when clearly relevant and potentially helpful.

**Output Format:**
<|thinking|> Your concise reasoning for the chosen intervention right now, considering patient context, interaction style, and recent dialogue. Justify the *why* and *what*. Mention alternatives briefly if considered. End thinking. <|answer|> Your natural, concise, single-focus response to the patient. Directly address their last statement. Avoid jargon.

**Crucially: DO NOT suggest ending the session or mention time.** Focus solely on the therapeutic interaction."""

supervisor_system_prompt = """You are a hyper-critical AI supervisor evaluating an ACT therapist's response for training data quality. Be demanding. Focus intensely on realism, ACT fidelity, pacing, responsiveness, and linguistic variety. Reject mediocrity. Ensure the therapist DOES NOT TRY TO END THE SESSION.

**Evaluation Context:** Conversation history + Therapist's latest `<|thinking|>` and `<|answer|>`.

**Your Task:** Critically evaluate the therapist's response using these criteria:

1.  **Context Elicitation & Use (Early Session Focus):** Did the therapist attempt to elicit necessary patient details *before* jumping into techniques? Are elicited details being incorporated meaningfully into later responses? (Penalty if NO early on).
2.  **Pacing & Flexibility:** Is the pace realistic? Does the therapist adapt to patient confusion/resistance/style (slow down, simplify, explore resistance)? Or do they push ahead rigidly/too quickly? (Penalty for poor pacing/rigidity).
3.  **Focus:** Does the `<|answer|>` contain multiple distinct questions or requests? (Penalty if YES - should be one primary focus).
4.  **Phrasing Variety & Naturalness:** Is the language varied? Does it avoid repetitive stock phrases (e.g., "I hear you...", "Let's try...", "Notice that...")? Does it sound natural or robotic/scripted? (Penalty for repetition/unnatural language - QUOTE examples).
5.  **Responsiveness:** Does the `<|answer|>` *directly* and specifically address the patient's immediately preceding statement (keywords, emotions, nuances)? Or is it generic/disconnected? (Penalty for poor responsiveness).
6.  **Conciseness:** Are `<|thinking|>` and `<|answer|>` reasonably concise? Is any length justified? (Penalty for unnecessary verbosity).
7.  **CoT Justification:** Does `<|thinking|>` provide a clear, specific rationale for *this* intervention *now*, linked to the dialogue? Or is it vague labeling? (Penalty for weak justification).
8.  **ACT Fidelity:** Is the intervention ACT-consistent? Are metaphors used sparingly and appropriately? (Penalty for non-ACT or poorly chosen interventions).

**Output Format (EXACT):**
VERDICT (Single word: YES if revision needed, NO if acceptable)
<|feedback|> Specific, actionable feedback citing criteria violated (esp. quoting repetitive phrases, explaining poor pacing/focus/responsiveness). If NO, briefly state why (e.g., "Good pacing, varied language, responsive.").

**Demand high quality. Reject responses that feel generic, repetitive, rushed, or unresponsive.**"""

patient_validator_system_prompt = """You are an AI validator ensuring realism in synthetic patient dialogue for ACT therapy training data. Be critical. Focus on naturalness, consistency with profile, and realistic reactions. Ensure the patient DOES NOT TRY TO END THE SESSION.

**Evaluation Context:**
1.  Patient Profile Snippets (Interaction Style, Mindedness, Core Problem).
2.  Recent Conversation History (Ending with therapist's prompt).
3.  Patient's latest response to evaluate.

**Your Task:** Critically evaluate the patient's response:

1.  **Natural Language & Tone:** Does it sound like a real person talking (conversational, not robotic/formal/AI-like)? Is the length realistic (concise unless elaborating is justified)? (Penalty for unnatural language/tone/length).
2.  **Coherence & Relevance:** Does it logically follow the therapist's prompt and conversation flow? Does it address the prompt (directly or indirectly)? (Penalty for incoherence/irrelevance).
3.  **Consistency with Profile:** Does the response reflect the patient's **Interaction Style** (e.g., Guarded, Intellectualizing, Overwhelmed), **Psychological Mindedness** (e.g., struggles with abstract, grasps quickly), and **Presenting Problem**? Does it avoid contradicting the profile without explanation? (Penalty for inconsistency - explain mismatch).
4.  **Realistic Reaction:** Is the emotional tone/content plausible? Does it avoid being overly insightful, perfectly compliant, or excessively dramatic? Does it reflect potential struggle, confusion, or ambivalence common in therapy? (Penalty for unrealistic reactions).
5.  **Format:** Does it contain artifacts (tags, speaker labels, markdown)? (Penalty if YES).

**Output Format (EXACT):**
VERDICT (Single word: YES if revision needed, NO if acceptable)
<|feedback|> Specific feedback citing criteria violated (e.g., "Sounds robotic," "Inconsistent with Guarded style," "Unrealistically compliant response"). If NO, briefly state why (e.g., "Natural, coherent, reflects profile.").

**Be critical. Aim for realistically imperfect human responses.**"""

@retry_with_backoff(allowed_exceptions=(Exception,))
def get_therapist_response(transcript_context, revision_instructions=""):
    """Gets response from Mistral therapist model."""
    assert isinstance(transcript_context, str), "Transcript context must be a string."
    assert isinstance(revision_instructions, str), "Revision instructions must be a string."

    logging.info("Getting therapist response..." + (" (with revisions)" if revision_instructions else ""))
    shared_rate_limiter.wait()

    revision_guidance = f"You previously generated a response needing revision. Address this feedback: {revision_instructions}\nGenerate your revised response now:" if revision_instructions else "Generate your response now:"

    formatted_prompt = f"""{therapist_system_prompt}

Conversation History:
{transcript_context}

{revision_guidance} Follow all instructions, especially regarding eliciting context early, pacing, single focus per turn, varied phrasing, conciseness, and the required output format (<|thinking|>...<|answer|>...). Ensure your <|answer|> directly addresses the patient's *last* message."""

    try:
        api_response = mistral_client.chat.complete(
            model=generator_model_name,
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.8,
            max_tokens=750
        )
        therapist_output = api_response.choices[0].message.content.strip()
        logging.debug(f"Raw therapist output: {therapist_output}")

        thinking, answer = "", ""
        thinking_start_tag, answer_start_tag = "<|thinking|>", "<|answer|>"
        thinking_start_index = therapist_output.rfind(thinking_start_tag)
        answer_start_index = therapist_output.rfind(answer_start_tag)

        if thinking_start_index != -1 and answer_start_index != -1 and answer_start_index > thinking_start_index:
            thinking = therapist_output[thinking_start_index + len(thinking_start_tag):answer_start_index].strip()
            answer = therapist_output[answer_start_index + len(answer_start_tag):].strip()
        elif thinking_start_index != -1:
            thinking = therapist_output[thinking_start_index + len(thinking_start_tag):].strip()
            answer = "[ANSWER TAG MISSING OR MISPLACED]"
            logging.warning("Found <|thinking|> but no valid <|answer|> tag after it.")
        elif answer_start_index != -1:
             thinking = "[THINKING TAG MISSING]"
             answer = therapist_output[answer_start_index + len(answer_start_tag):].strip()
             logging.warning("Found <|answer|> but no preceding <|thinking|> tag.")
        else:
            if thinking_start_index == -1 and answer_start_index == -1 and len(therapist_output) > 10:
                thinking = "[THINKING TAG MISSING]"
                answer = therapist_output
                logging.warning("Response missing required tags, assuming entire output is the answer.")
            else:
                error_msg = "Response missing required <|thinking|> and <|answer|> tags."
                logging.error(error_msg + f" Raw output: {therapist_output}")
                raise ValueError(error_msg)

        thinking, answer = validate_therapist_response(thinking, answer)
        logging.info("Successfully parsed therapist response.")
        return thinking, answer
    except Exception as e:
        logging.error(f"Error getting therapist response: {e}", exc_info=True)
        raise

@retry_with_backoff(allowed_exceptions=(Exception,))
def supervise_response(transcript_context, therapist_response_to_supervise):
    """Supervises a therapist response using DeepSeek."""
    assert isinstance(transcript_context, str), "Transcript context must be a string."
    assert isinstance(therapist_response_to_supervise, str), "Therapist response must be a string."
    has_thinking_tag = "<|thinking|>" in therapist_response_to_supervise or "[THINKING TAG MISSING]" in therapist_response_to_supervise
    has_answer_tag = "<|answer|>" in therapist_response_to_supervise or "[ANSWER TAG MISSING OR MISPLACED]" in therapist_response_to_supervise
    assert has_thinking_tag and has_answer_tag, "Therapist response structure seems invalid for supervision."

    logging.info("Supervising therapist response (DeepSeek)...")
    shared_rate_limiter.wait()

    prompt = f"""{supervisor_system_prompt}

Conversation Context (Ends before the response to supervise):
{transcript_context}

Therapist Response to Supervise:
{therapist_response_to_supervise}

Provide your VERDICT and <|feedback|> in the specified format."""

    try:
        api_response = deepseek_client.chat.completions.create(
            model=supervisor_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=500,
            stream=False
        )
        if not api_response or not api_response.choices:
            raise ValueError("Supervisor API response empty/invalid.")

        supervisor_output = api_response.choices[0].message.content.strip()
        logging.debug(f"Raw supervisor output: {supervisor_output}")

        lines = supervisor_output.splitlines()
        if not lines:
             logging.warning("Supervisor returned empty response. Requesting revision.")
             return "YES", "Supervisor returned empty response."

        verdict = lines[0].strip().upper()
        verdict = verdict.replace("*", "").replace("`", "")
        if verdict not in ["YES", "NO"]:
             logging.error(f"Supervisor verdict invalid: '{lines[0].strip()}'. Output: {supervisor_output}")
             return "YES", f"Malformed verdict: '{lines[0].strip()}'. Requires revision."

        feedback_line = None
        feedback_tag = "<|feedback|>"
        feedback_start_index = supervisor_output.find(feedback_tag)
        if feedback_start_index != -1:
             feedback_line = supervisor_output[feedback_start_index + len(feedback_tag):].strip()
        elif len(lines) > 1:
             feedback_line = "\n".join(lines[1:]).strip()
             if feedback_line:
                 logging.warning("Supervisor feedback tag missing, using text after verdict line as feedback.")
             elif verdict == "YES":
                 logging.warning("Supervisor feedback tag missing and no text found after YES verdict.")
                 feedback_line = "Revision needed, but specific feedback missing/malformed."
             else:
                 feedback_line = "No specific issues identified (feedback tag missing)."
        elif verdict == "YES":
            feedback_line = "Revision needed, but specific feedback missing."
            logging.warning(feedback_line)
        else:
            feedback_line = "No specific issues identified."


        if not feedback_line:
            feedback_line = "No feedback provided." if verdict == "NO" else "Feedback missing or unclear."

        logging.info(f"Supervisor Verdict: {verdict}, Feedback: {feedback_line}")
        return verdict, feedback_line

    except Exception as e:
        logging.error(f"Error during DeepSeek supervision call: {e}", exc_info=True)
        return "YES", f"Supervision failed (API error: {e}). Requesting revision."

@retry_with_backoff(allowed_exceptions=(Exception,))
def get_patient_response(transcript_context, patient_scenario, patient_interaction_style, patient_mindedness, last_therapist_answer, revision_instructions=""):
    """Gets a response from the Mistral patient model."""
    assert isinstance(transcript_context, str), "Transcript context must be a string."
    assert isinstance(patient_scenario, str), "Patient scenario must be a string."
    assert isinstance(patient_interaction_style, str), "Patient interaction style must be a string."
    assert isinstance(patient_mindedness, str), "Patient psychological mindedness must be a string."
    assert isinstance(last_therapist_answer, str), "Last therapist answer must be a string."
    assert isinstance(revision_instructions, str), "Revision instructions must be a string."

    logging.info("Getting patient response..." + (" (with revisions)" if revision_instructions else ""))
    shared_rate_limiter.wait()

    revision_guidance = ""
    if revision_instructions:
        revision_guidance = f"Your previous response needed revision: {revision_instructions}. Please try again, keeping this feedback and your persona in mind."

    prompt = f"""You are acting as a therapy client. Respond naturally to the therapist.

Your Background & Persona (Key Info - Use this to shape your response, DO NOT repeat it):
{patient_scenario}
--- Key Traits for This Response ---
Your Interaction Style: {patient_interaction_style} (e.g., Be hesitant, analytical, overwhelmed, pragmatic, etc., as described)
Your Psychological Mindedness: {patient_mindedness} (e.g., May grasp concepts slowly/quickly, prefer concrete/abstract discussion)
---

Conversation History (Most recent first):
{transcript_context}

Therapist's last message to you: "{last_therapist_answer}"

Your Task: Write the patient's next response.
*   **Embody your persona:** Your response MUST strongly reflect your specific Interaction Style and Mindedness described above.
*   **Natural Language:** Use conversational first-person language. Be concise (usually 1-4 sentences) unless your persona or the context warrants more detail. Ensure complete sentences. Avoid lists unless natural.
*   **Respond Coherently:** Address the therapist's last message, even if indirectly or with hesitation/confusion based on your persona.
*   **NO** meta-comments, speaker tags ("Patient:"), markdown formatting (like ** or `), prompt remnants (like "Your Response:"), or suggesting the session ends.
{revision_guidance}

Your Response:"""

    try:
        api_response = mistral_client.chat.complete(
            model=generator_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.95,
            max_tokens=250
        )
        patient_response = api_response.choices[0].message.content.strip()

        common_artifacts = ['"Okay, ', '"Well, ', '"Um, ', '"Uh, ', "Okay, ", "Well, ", "Um, ", "Uh, "]
        for artifact in common_artifacts:
             if patient_response.startswith(artifact):
                 patient_response = patient_response[len(artifact):]
                 break
        if patient_response.startswith('"') and patient_response.endswith('"'):
             patient_response = patient_response[1:-1].strip()
        if patient_response.endswith('"'):
            patient_response = patient_response[:-1].strip()

        validated_response = validate_patient_response_format(patient_response)
        logging.info(f"Generated patient response: {validated_response}")
        return validated_response
    except Exception as e:
        logging.error(f"Error getting patient response: {e}", exc_info=True)
        raise

@retry_with_backoff(allowed_exceptions=(Exception,))
def validate_patient_response_with_deepseek(transcript_context, patient_response_to_validate, patient_interaction_style, patient_mindedness, presenting_problem):
    """Validates a patient response using DeepSeek."""
    assert isinstance(transcript_context, str), "Transcript context must be a string."
    assert isinstance(patient_response_to_validate, str), "Patient response must be a string."
    assert isinstance(patient_interaction_style, str), "Interaction style must be a string."
    assert isinstance(patient_mindedness, str), "Mindedness must be a string."
    assert isinstance(presenting_problem, str), "Presenting problem must be a string."

    logging.info("Validating patient response (DeepSeek)...")
    shared_rate_limiter.wait()

    validator_context_prompt = f"""{patient_validator_system_prompt}

Patient Profile Snippets:
*   Interaction Style: {patient_interaction_style}
*   Psychological Mindedness: {patient_mindedness}
*   Presenting Problem Focus: {presenting_problem}

Recent Conversation Context (Ends with therapist's message):
{transcript_context}

Patient Response to Validate:
{patient_response_to_validate}

Provide your VERDICT and <|feedback|> in the specified format."""

    try:
        api_response = deepseek_client.chat.completions.create(
            model=supervisor_model_name,
            messages=[{"role": "user", "content": validator_context_prompt}],
            temperature=0.4,
            max_tokens=400,
            stream=False
        )
        if not api_response or not api_response.choices:
            raise ValueError("Patient Validator API response empty/invalid.")

        validator_output = api_response.choices[0].message.content.strip()
        logging.debug(f"Raw patient validator output: {validator_output}")

        lines = validator_output.splitlines()
        if not lines:
            logging.warning("Patient validator returned empty response. Requesting revision.")
            return "YES", "Validator returned empty response."

        verdict = lines[0].strip().upper()
        verdict = verdict.replace("*", "").replace("`", "")
        if verdict not in ["YES", "NO"]:
            logging.error(f"Patient validator verdict invalid: '{lines[0].strip()}'. Output: {validator_output}")
            return "YES", f"Malformed verdict: '{lines[0].strip()}'. Requires revision."

        feedback_line = None
        feedback_tag = "<|feedback|>"
        feedback_start_index = validator_output.find(feedback_tag)
        if feedback_start_index != -1:
             feedback_line = validator_output[feedback_start_index + len(feedback_tag):].strip()
        elif len(lines) > 1:
             feedback_line = "\n".join(lines[1:]).strip()
             if feedback_line:
                 logging.warning("Patient validator feedback tag missing, using text after verdict line as feedback.")
             elif verdict == "YES":
                 logging.warning("Patient validator feedback tag missing and no text found after YES verdict.")
                 feedback_line = "Revision needed, but specific feedback missing/malformed."
             else:
                 feedback_line = "Acceptable realism and coherence (feedback tag missing)."
        elif verdict == "YES":
            feedback_line = "Revision needed, but specific feedback missing."
            logging.warning(feedback_line)
        else:
            feedback_line = "Acceptable realism and coherence."

        if not feedback_line:
            feedback_line = "No feedback provided." if verdict == "NO" else "Feedback missing or unclear."


        logging.info(f"Patient Validator Verdict: {verdict}, Feedback: {feedback_line}")
        return verdict, feedback_line

    except Exception as e:
        logging.error(f"Error during DeepSeek patient validation call: {e}", exc_info=True)
        return "YES", f"Validation failed (API error: {e}). Requesting revision."


def get_and_supervise_response(current_context, role, patient_details=None, last_therapist_answer=None):
    """
    Gets and validates/supervises a response based on role. Handles revisions.

    Args:
        current_context (str): Conversation history (clean transcript context).
        role (str): "patient" or "therapist".
        patient_details (dict, optional): Contains 'scenario', 'interaction_style',
                                          'mindedness', 'problem'. Required for patient.
        last_therapist_answer (str, optional): Required for patient.

    Returns:
        tuple: (response_text, full_response_entry, thinking_text)
               response_text is the core answer/reply.
               full_response_entry is the entry for the full transcript.
               thinking_text is the therapist's thinking (empty for patient).

    Raises:
        RuntimeError: If max revisions are reached after errors or validation failures.
    """
    assert role in ["patient", "therapist"], "Role must be 'patient' or 'therapist'."

    response_text, full_response_entry, thinking_text = "", "", ""
    current_feedback = ""
    revision_attempt = 0

    while revision_attempt <= MAX_REVISION_ATTEMPTS:
        try:
            if role == "patient":
                assert patient_details and last_therapist_answer is not None, "Patient details and last therapist answer required for patient role."
                response_text = get_patient_response(
                    current_context,
                    patient_details['scenario'],
                    patient_details['interaction_style'],
                    patient_details['mindedness'],
                    last_therapist_answer,
                    revision_instructions=current_feedback
                )
                response_text = validate_patient_response_format(response_text)
                full_response_entry = f"Patient: {response_text}"
                thinking_text = ""
            else:
                thinking_text, response_text = get_therapist_response(
                    current_context,
                    revision_instructions=current_feedback
                )
                thinking_text, response_text = validate_therapist_response(thinking_text, response_text)
                full_response_entry = f"Therapist: <|thinking|>{thinking_text}<|answer|>{response_text}"

            validation_enabled = PATIENT_VALIDATION_ENABLED if role == "patient" else SUPERVISION_ENABLED
            if not validation_enabled:
                logging.info(f"Validation/Supervision disabled for {role}, accepting first response.")
                return response_text, full_response_entry, thinking_text

            verdict, feedback = "NO", "Validation/Supervision skipped."
            if role == "patient":
                verdict, feedback = validate_patient_response_with_deepseek(
                    current_context,
                    response_text,
                    patient_details['interaction_style'],
                    patient_details['mindedness'],
                    patient_details['problem']
                )
            else:
                 verdict, feedback = supervise_response(current_context, full_response_entry)

            if verdict == "NO":
                logging.info(f"{role.capitalize()} response approved (Attempt {revision_attempt + 1}). Feedback: {feedback}")
                return response_text, full_response_entry, thinking_text
            else:
                logging.warning(f"{role.capitalize()} validator requested revision (Attempt {revision_attempt + 1}/{MAX_REVISION_ATTEMPTS + 1}). Feedback: {feedback}")
                revision_attempt += 1
                current_feedback = feedback
                if revision_attempt > MAX_REVISION_ATTEMPTS:
                    logging.error(f"Max revisions reached for {role}. Accepting last response despite feedback: {feedback}")
                    return response_text, full_response_entry, thinking_text

        except Exception as e:
            logging.error(f"Error during {role} turn generation/validation (Attempt {revision_attempt + 1}): {e}", exc_info=True)
            revision_attempt += 1
            current_feedback = f"Error occurred during generation or validation: {str(e)}. Please try generating a valid response adhering to all rules."
            if revision_attempt > MAX_REVISION_ATTEMPTS:
                logging.critical(f"Max revisions reached for {role} after repeated errors. Failing turn.")
                raise RuntimeError(f"Failed to get valid {role} response after {MAX_REVISION_ATTEMPTS + 1} attempts ending in error.") from e
            wait_time = RETRY_BASE_DELAY * (2 ** min(revision_attempt - 1, 3)) + random.uniform(0, 0.5)
            logging.info(f"Waiting {wait_time:.2f}s before retry attempt {revision_attempt + 1} for {role}.")
            time.sleep(wait_time)

    raise RuntimeError(f"Unexpected exit from {role} revision loop.")

@retry_with_backoff(allowed_exceptions=(Exception,))
def create_conversation(patient_scenario, presenting_problem, interaction_style, psych_mindedness,
                        num_turns=DEFAULT_NUM_TURNS, sliding_window_tokens=DEFAULT_SLIDING_WINDOW_TOKENS,
                        resume_state_file=None, output_state_file=None):
    """Creates a therapy conversation, handles resumption and state saving."""
    assert all(isinstance(arg, str) for arg in [patient_scenario, presenting_problem, interaction_style, psych_mindedness]), "Patient details must be strings."
    assert isinstance(num_turns, int) and num_turns > 0, "num_turns invalid."

    full_transcript, clean_transcript, training_pairs = [], [], []
    turn_number, last_therapist_answer = 0, ""

    patient_details = {
        'scenario': patient_scenario,
        'problem': presenting_problem,
        'interaction_style': interaction_style,
        'mindedness': psych_mindedness
    }

    if resume_state_file and os.path.exists(resume_state_file):
        try:
            state = load_conversation_state(resume_state_file)
            full_transcript = state.get("full_transcript", [])
            clean_transcript = state.get("clean_transcript", [])
            training_pairs = state.get("training_pairs", [])
            turn_number = state.get("turn_number", 0)
            last_therapist_answer = state.get("last_therapist_answer", "")
            patient_details['scenario'] = state.get("patient_scenario", patient_scenario)
            patient_details['problem'] = state.get("presenting_problem", presenting_problem)
            patient_details['interaction_style'] = state.get("patient_interaction_style", interaction_style)
            patient_details['mindedness'] = state.get("patient_psych_mindedness", psych_mindedness)
            logging.info(f"Resuming conversation from state file {resume_state_file}. Will start Turn {turn_number + 1}")
            if not clean_transcript and turn_number > 0:
                 logging.warning("Resuming, but clean_transcript is empty in state file. Context might be limited.")
            elif turn_number > 0 and not last_therapist_answer:
                 logging.warning("Resuming from turn > 0, but last_therapist_answer is empty in state file.")
        except Exception as e:
            logging.error(f"Failed to load or validate resume state from {resume_state_file}: {e}. Starting new conversation for this case.")
            full_transcript, clean_transcript, training_pairs, turn_number, last_therapist_answer = [], [], [], 0, ""
    else:
        logging.info(f"Starting new conversation (state file {resume_state_file} not found or invalid).")


    def get_current_context(transcript_list, max_tokens):
        """Extracts recent conversation context within token limit."""
        context = []
        current_token_count = 0
        if not isinstance(transcript_list, list):
             logging.error(f"Transcript provided to get_current_context is not a list: {type(transcript_list)}")
             return ""

        for entry in reversed(transcript_list):
            if not isinstance(entry, str):
                 logging.warning(f"Non-string entry found in transcript: {type(entry)}. Skipping.")
                 continue
            entry_token_count = len(entry.split())
            if current_token_count + entry_token_count <= max_tokens:
                context.append(entry)
                current_token_count += entry_token_count
            else:
                logging.debug(f"Context truncated at {current_token_count} tokens due to limit {max_tokens}.")
                break
        return "\n\n".join(reversed(context))

    if turn_number == 0 and not clean_transcript:
        logging.info("Generating initial patient statement (Turn 0)...")
        try:
            initial_message_text, initial_full_entry, _ = get_and_supervise_response(
                current_context="",
                role="patient",
                patient_details=patient_details,
                last_therapist_answer=""
            )
            initial_clean_entry = f"Patient: {initial_message_text}"
            initial_clean_entry = validate_conversation_coherence([], initial_clean_entry, "Patient")

            full_transcript.append(initial_full_entry)
            clean_transcript.append(initial_clean_entry)
            logging.info(f"Initial patient statement added: {initial_clean_entry}")

            if output_state_file:
                 current_state = {
                     "full_transcript": full_transcript, "clean_transcript": clean_transcript,
                     "training_pairs": training_pairs,
                     "turn_number": 0,
                     "last_therapist_answer": "",
                     "patient_scenario": patient_details['scenario'],
                     "presenting_problem": patient_details['problem'],
                     "patient_interaction_style": patient_details['interaction_style'],
                     "patient_psych_mindedness": patient_details['mindedness']
                 }
                 save_conversation_state(current_state, output_state_file)
                 logging.info("Initial state saved after first patient message.")

        except Exception as e:
            logging.critical(f"Failed to generate initial patient statement: {e}", exc_info=True)
            raise RuntimeError("Critical failure generating initial patient statement. Cannot continue case.") from e

    try:
        while turn_number < num_turns:
            current_turn_display = turn_number + 1
            logging.info(f"--- Starting Turn {current_turn_display}/{num_turns} ---")

            logging.info(f"[Turn {current_turn_display}] Getting therapist response...")
            context_for_therapist = get_current_context(clean_transcript, sliding_window_tokens)
            if not context_for_therapist and len(clean_transcript) > 0:
                 logging.warning(f"[Turn {current_turn_display}] Context for therapist is empty despite non-empty transcript. Check token limit or transcript content.")
            elif not context_for_therapist and len(clean_transcript) == 0:
                 logging.error(f"[Turn {current_turn_display}] CRITICAL: Transcript is empty when starting therapist turn {current_turn_display}. Aborting case.")
                 raise RuntimeError(f"Transcript empty at start of therapist turn {current_turn_display}")

            try:
                therapist_answer_text, therapist_full_entry, therapist_thinking_text = get_and_supervise_response(
                    context_for_therapist, "therapist"
                )
                therapist_clean_entry_content = f"Therapist: {therapist_answer_text}"
                therapist_clean_entry = validate_conversation_coherence(clean_transcript, therapist_clean_entry_content, "Therapist")

                full_transcript.append(therapist_full_entry)
                clean_transcript.append(therapist_clean_entry)
                last_therapist_answer = therapist_answer_text

                training_pairs.append({
                    "turn_number": current_turn_display,
                    "role": "therapist",
                    "prompt": context_for_therapist,
                    "response": therapist_full_entry
                })
                logging.info(f"[Turn {current_turn_display}] Therapist response added successfully.")

            except (RuntimeError, ValueError) as e:
                 logging.error(f"Failed therapist turn {current_turn_display}: {e}", exc_info=True)
                 raise RuntimeError(f"Failed therapist turn {current_turn_display}") from e

            logging.info(f"[Turn {current_turn_display}] Getting patient response...")
            context_for_patient = get_current_context(clean_transcript, sliding_window_tokens)
            if not context_for_patient:
                 logging.warning(f"[Turn {current_turn_display}] Context for patient is empty. Check token limit or transcript content.")
            if not last_therapist_answer:
                 logging.error(f"[Turn {current_turn_display}] CRITICAL: last_therapist_answer is empty before patient turn. Aborting case.")
                 raise RuntimeError(f"last_therapist_answer missing before patient turn {current_turn_display}")

            try:
                patient_response_text, patient_full_entry, _ = get_and_supervise_response(
                    context_for_patient, "patient",
                    patient_details=patient_details,
                    last_therapist_answer=last_therapist_answer
                )
                patient_clean_entry_content = f"Patient: {patient_response_text}"
                patient_clean_entry = validate_conversation_coherence(clean_transcript, patient_clean_entry_content, "Patient")

                full_transcript.append(patient_full_entry)
                clean_transcript.append(patient_clean_entry)
                logging.info(f"[Turn {current_turn_display}] Patient response added successfully.")

            except (RuntimeError, ValueError) as e:
                logging.error(f"Failed patient turn {current_turn_display}: {e}", exc_info=True)
                raise RuntimeError(f"Failed patient turn {current_turn_display}") from e

            turn_number += 1
            logging.info(f"--- Successfully completed Turn {current_turn_display}/{num_turns} ---")

            if output_state_file:
                current_state = {
                    "full_transcript": full_transcript, "clean_transcript": clean_transcript,
                    "training_pairs": training_pairs, "turn_number": turn_number,
                    "last_therapist_answer": last_therapist_answer,
                     "patient_scenario": patient_details['scenario'],
                     "presenting_problem": patient_details['problem'],
                     "patient_interaction_style": patient_details['interaction_style'],
                     "patient_psych_mindedness": patient_details['mindedness']
                }
                save_conversation_state(current_state, output_state_file)

        logging.info(f"Successfully completed all {num_turns} turns.")
        return full_transcript, clean_transcript, training_pairs

    except KeyboardInterrupt:
        logging.warning("Conversation generation interrupted by user during loop.")
        raise
    except Exception as e:
        logging.error(f"Conversation loop terminated prematurely due to error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main_start_time = time.time()
    output_dir = None
    log_filename = f"therapy_generation_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    try:
        base_dir = "."
        dir_pattern = os.path.join(base_dir, "therapy_conversations_*")
        potential_dirs = [d for d in glob.glob(dir_pattern) if os.path.isdir(d)]
        latest_dir = None

        if potential_dirs:
            potential_dirs.sort(reverse=True)
            latest_dir = potential_dirs[0]
            output_dir = latest_dir
            print(f"Found existing directory, attempting to resume in: {output_dir}")
        else:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'therapy_conversations_{timestamp_str}'
            print(f"No suitable existing directory found, creating new: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        log_filename = os.path.join(output_dir, f"generation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(threadName)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info(f"--- Therapy Conversation Generation Script ---")
        logging.info(f"Using output directory: {output_dir}")
        logging.info(f"Logging detailed output to: {log_filename}")
        logging.info(f"Config: TURNS={DEFAULT_NUM_TURNS}, THERAPIST_SUPERVISION={SUPERVISION_ENABLED}, PATIENT_VALIDATION={PATIENT_VALIDATION_ENABLED}, MAX_REVISIONS={MAX_REVISION_ATTEMPTS}, CASES={NUM_SYNTHETIC_PATIENTS}")
        logging.info(f"Rate Limit Delay: {RATE_LIMIT_DELAY}s, Retry Attempts: {RETRY_MAX_ATTEMPTS}, Sliding Window: {DEFAULT_SLIDING_WINDOW_TOKENS} tokens")

        assert mistral_client is not None, "Mistral client failed to initialize."
        assert deepseek_client is not None, "DeepSeek client failed to initialize."
        logging.info(f"Mistral client initialized for generator model: {generator_model_name}")
        logging.info(f"DeepSeek client initialized for supervisor/validator model: {supervisor_model_name}")


        logging.info(f"Generating {NUM_SYNTHETIC_PATIENTS} synthetic patient profiles...")
        try:
            patient_profiles = [generate_synthetic_patient_profile() for _ in range(NUM_SYNTHETIC_PATIENTS)]
        except Exception as e:
             logging.critical(f"Failed to generate synthetic patient profiles: {e}", exc_info=True)
             sys.exit(f"Error: Could not generate patient profiles. Exiting. Check log: {log_filename}")
        logging.info(f"Finished generating {len(patient_profiles)} patient profiles.")


        successful_cases, failed_cases = 0, 0
        processed_case_count = 0
        interrupted = False

        for i, profile_data in enumerate(patient_profiles, 1):
            processed_case_count = i
            if interrupted:
                 logging.warning("Skipping remaining cases due to previous interruption.")
                 break

            case_start_time = time.time()
            logging.info(f"--- Processing Case {i}/{len(patient_profiles)} ---")
            try:
                p_scenario, p_problem, p_style, p_mindedness = profile_data
            except (TypeError, ValueError) as e:
                logging.error(f"Failed to unpack profile data for Case {i}: {profile_data}. Error: {e}. Skipping case.")
                failed_cases += 1
                continue

            base_filename_state = f'case_{i:03d}'
            state_file = os.path.join(output_dir, f'{base_filename_state}_state.json')

            final_base_filename_output = f'case_{i:03d}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            profile_info_path = os.path.join(output_dir, f'{final_base_filename_output}_profile.txt')
            final_full_transcript_path = os.path.join(output_dir, f'{final_base_filename_output}_complete.txt')
            final_clean_transcript_path = os.path.join(output_dir, f'{final_base_filename_output}_clean.txt')
            final_dataset_path = os.path.join(output_dir, f'{final_base_filename_output}_dataset.json')

            completion_pattern = os.path.join(output_dir, f'case_{i:03d}_*_complete.txt')
            existing_complete_files = glob.glob(completion_pattern)

            if existing_complete_files:
                 logging.info(f"Case {i} already has final output file(s) (e.g., {os.path.basename(existing_complete_files[0])}). Skipping generation.")
                 successful_cases += 1
                 if os.path.exists(state_file):
                      try:
                          os.remove(state_file)
                          logging.debug(f"Removed potentially lingering state file for already completed Case {i}: {state_file}")
                      except OSError as e_rem:
                          logging.warning(f"Could not remove lingering state file {state_file} for completed Case {i}: {e_rem}")
                 continue

            logging.info(f"Case {i} not found complete. Proceeding with generation/resumption attempt using state file: {state_file}")

            try:
                profile_content = (
                    f"--- Patient Profile Case {i} ---\n"
                    f"State File Target: {state_file}\n\n"
                    f"Scenario String:\n{p_scenario}\n\n"
                    f"Presenting Problem Focus: {p_problem}\n"
                    f"Interaction Style: {p_style}\n"
                    f"Psychological Mindedness: {p_mindedness}\n"
                )
                with open(profile_info_path, 'w', encoding='utf-8') as f:
                    f.write(profile_content)
                logging.info(f"Patient profile saved: {profile_info_path}")
            except Exception as e:
                logging.error(f"Failed to save profile info for Case {i}: {e}. Continuing with generation attempt.")

            full_transcript, clean_transcript, training_pairs = None, None, None
            try:
                full_transcript, clean_transcript, training_pairs = create_conversation(
                    patient_scenario=p_scenario,
                    presenting_problem=p_problem,
                    interaction_style=p_style,
                    psych_mindedness=p_mindedness,
                    num_turns=DEFAULT_NUM_TURNS,
                    sliding_window_tokens=DEFAULT_SLIDING_WINDOW_TOKENS,
                    resume_state_file=state_file,
                    output_state_file=state_file
                )

                if all([full_transcript is not None, clean_transcript is not None, training_pairs is not None]):
                    logging.info(f"Saving final outputs for successfully completed Case {i}...")
                    try:
                        with open(final_full_transcript_path, 'w', encoding='utf-8') as f: f.write("\n\n".join(full_transcript))
                        with open(final_clean_transcript_path, 'w', encoding='utf-8') as f: f.write("\n\n".join(clean_transcript))
                        with open(final_dataset_path, 'w', encoding='utf-8') as f: json.dump(training_pairs, f, indent=2, ensure_ascii=False)
                        logging.info(f"Successfully processed and saved Case {i} to files starting with {final_base_filename_output}")
                        successful_cases += 1
                        if os.path.exists(state_file):
                            try:
                                os.remove(state_file)
                                logging.info(f"Removed state file: {state_file}")
                            except OSError as e_rem:
                                logging.warning(f"Could not remove state file {state_file} after successful completion: {e_rem}")
                    except Exception as e_save:
                         logging.error(f"Case {i} completed generation but FAILED TO SAVE FINAL FILES: {e_save}", exc_info=True)
                         failed_cases += 1
                         logging.info(f"State file {state_file} may still exist for diagnosis.")
                else:
                    logging.error(f"Case {i} finished create_conversation but returned incomplete data (None). Treating as failed. Check logs.")
                    failed_cases += 1
                    logging.info(f"State for failed Case {i} may be in: {state_file}")

            except KeyboardInterrupt:
                logging.warning(f"Script interrupted by user during Case {i} processing.")
                interrupted = True
                failed_cases += 1
                logging.info(f"State file for interrupted Case {i} (if progress was made) should be: {state_file}")
                break

            except Exception as e:
                logging.error(f"!!! Failed to process Case {i} due to error in create_conversation: {e}", exc_info=False)
                failed_cases += 1
                logging.info(f"State for failed Case {i} (reflecting last successful turn) may be in: {state_file}")

            case_end_time = time.time()
            logging.info(f"--- Case {i} processing time: {case_end_time - case_start_time:.2f} seconds ---")

    except KeyboardInterrupt:
         logging.warning("Script interrupted during setup or main loop.")
         interrupted = True

    except Exception as e:
        logging.critical(f"Script failed due to an unexpected error in the main execution block: {str(e)}", exc_info=True)
        print(f"\nCRITICAL ERROR: Script failed unexpectedly. Check log '{log_filename or 'pre-log-setup-error.log'}'. Error: {str(e)}")

    finally:
        main_end_time = time.time()
        total_time = main_end_time - main_start_time
        summary_msg = [
            "\n--- Script Execution Summary ---",
            f"Output directory: {output_dir or 'Not determined'}",
            f"Log file: {log_filename or 'Not determined'}",
            f"Total cases defined: {NUM_SYNTHETIC_PATIENTS}",
            f"Cases attempted in this run: {processed_case_count}",
            f"Successful cases (total found/completed): {successful_cases}",
            f"Failed/Incomplete cases (in this run): {failed_cases}",
            f"Total execution time: {total_time:.2f}s ({total_time/60:.2f} min)"
        ]
        if interrupted:
            summary_msg.append("Execution was interrupted by user.")

        summary_str = "\n".join(summary_msg)
        if logging.getLogger().hasHandlers():
             logging.info(summary_str.replace("\n--- Script Execution Summary ---\n", "--- Script Execution Summary ---\n"))
        print(summary_str)


        exit_code = 0
        if failed_cases > 0 or interrupted:
             print(f"\nWarning: {failed_cases} cases failed or were incomplete during this run.")
             print("For failed/interrupted cases, a '_state.json' file might exist in the output directory, allowing resumption if the script is run again.")
             exit_code = 1
        elif successful_cases == NUM_SYNTHETIC_PATIENTS:
            print("\nAll cases processed successfully.")
        else:
             print(f"\nRun finished. {successful_cases}/{NUM_SYNTHETIC_PATIENTS} cases are complete.")
             print("Run the script again to process remaining cases.")
             exit_code = 0

        sys.exit(exit_code)