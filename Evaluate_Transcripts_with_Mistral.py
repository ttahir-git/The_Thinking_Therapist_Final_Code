import os
import re
import time
import logging
import pandas as pd
from mistralai import Mistral
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MISTRAL_API_KEY = ""
client = Mistral(api_key=MISTRAL_API_KEY)

LLM_JUDGE_MODEL = "ft:mistral-large-latest:9eab5258:20250703:2a2a1705"
LLM_CLEANUP_MODEL = "mistral-large-latest"
RATE_LIMIT_DELAY = 2.0
BACKOFF_BASE_DELAY = 5.0
MAX_BACKOFF_DELAY = 300.0

ORIGINAL_INPUT_DIRS = ['simulation_outputs_llama3_3b_instruct', 'simulation_outputs_orpo', 'simulation_outputs_orpo_no_COT', 'simulation_outputs_sft', 'simulation_outputs_sft_answer_only', 'simulation_outputs_llama3_3b_instruct_no_COT']

OUTPUT_DIR = "final_evaluations_mistral_sequential_Aug_19"
OUTPUT_RESULTS_CSV = os.path.join(OUTPUT_DIR, 'evaluation_results_with_parsing_health.csv')
FULL_EVALS_DIR = os.path.join(OUTPUT_DIR, 'full_text_evaluations')
PARSING_ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, 'parsing_errors.log')

INSTRUCTION_PROMPT = """I want you to act as an expert clinical supervisor specializing in Acceptance and Commitment Therapy (ACT). Your evaluation must go beyond surface-level techniques and focus on the underlying process and function of the therapist's interventions.
Core Principles for Evaluation:
Before rating, you must ground your analysis in these core principles of ACT fidelity:
Function Over Form: Your primary task is to evaluate the function of the therapist's interventions, not just their superficial form. For any given technique (e.g., mindfulness, grounding), constantly ask: "Is this being used to help the client make room for and be present with an experience, or is it being used as a tool to get rid of, control, or escape an experience?" Interventions aimed at control or avoidance, even if they sound like ACT, should be scored down.
Experiential vs. Conceptual: Be critical of therapy that remains at a purely conceptual, intellectual, or procedural level (i.e., talking about feelings and techniques). Reward therapy that is consistently experiential, guiding the client toward direct, in-the-moment contact with their thoughts, feelings, and bodily sensations as they are.
Focus on Workability: A skilled ACT therapist consistently orients the conversation around workability. The question is not whether a feeling is "bad" or a thought is "irrational," but whether the client's response to that feeling or thought is helping them move toward the life they want to live. Judge interventions based on whether they help the client explore the workability of their own behaviors.
Your Task:
Using this expert supervisor lens, rate the following transcript. Be very critical and adhere to the principles outlined above. Fill out the provided forms by adding numbers only, keeping the formatting exactly the same. Your final ratings should reflect a deep understanding of ACT process, not just a checklist of techniques. Just fill it out by adding numbers. Keep the formatting of the form the exact same. Remember rate the whole transcript, coherence, flow, ability to adapt and meeting the patient where they are at are all important."""

BLANK_FORM_TEMPLATE = """Output Form: ACT Fidelity Measure (ACT-FM)
Purpose: To evaluate a therapy transcript based on the Acceptance and Commitment Therapy Fidelity Measure (ACT-FM). The evaluation is based only on the therapist's observable behaviors in the transcript.
Rating Scale:
0: This behavior never occurred.
1: Therapist rarely enacts this behavior.
2: Therapist sometimes enacts this behavior.
3: Therapist consistently enacts this behavior.
ACT-FM Items to Rate:
Therapist Stance (Consistent)
item_1: Therapist chooses methods that are sensitive to the situation and context.
Rating: 
item_2: Therapist uses experiential methods/questions (helps client notice their own experience).
Rating: 
item_3: Therapist conveys that it is natural to experience painful thoughts and feelings.
Rating: 
item_4: Therapist demonstrates a willingness to sit with their own and the client's painful thoughts and feelings.
Rating: 
Therapist Stance (Inconsistent)
item_5: Therapist lectures the client (e.g., gives advice, tries to convince).
Rating: 
item_6: Therapist rushes to reassure, diminish or move on from "unpleasant" thoughts and feelings.
Rating: 
item_7: Therapist conversations are at an excessively conceptual level.
Rating: 
Open Response Style (Consistent)
item_8: Therapist helps the client to notice thoughts as separate experiences from the events they describe.
Rating: 
item_9: Therapist gives the client opportunities to notice how they interact with their thoughts and/or feelings.
Rating: 
item_10: Therapist encourages the client to "stay with" painful thoughts and feelings (in the service of their values).
Rating: 
Open Response Style (Inconsistent)
item_11: Therapist encourages the client to control or to diminish distress as the primary goal.
Rating: 
item_12: Therapist encourages the client to "think positive" or substitute thoughts as a treatment goal.
Rating: 
item_13: Therapist encourages the view that fusion or avoidance are implicitly bad, rather than judging them on workability.
Rating: 
Aware Response Style (Consistent)
item_14: Therapist uses present moment focus methods (e.g., mindfulness tasks, tracking).
Rating: 
item_15: Therapist helps the client to notice the stimuli that hook them away from the present moment.
Rating: 
item_16: Therapist helps the client to experience that they are bigger than their psychological experiences.
Rating: 
Aware Response Style (Inconsistent)
item_17: Therapist uses mindfulness/self-as-context methods as means to control or diminish unwanted experiences.
Rating: 
item_18: Therapist uses mindfulness/self-as-context methods to challenge the accuracy of beliefs or thoughts.
Rating: 
item_19: Therapist introduces mindfulness/self-as-context methods as formulaic exercises.
Rating: 
Engaged Response Style (Consistent)
item_20: Therapist gives the client opportunities to notice workable and unworkable responses.
Rating: 
item_21: Therapist gives the client opportunities to clarify their own values.
Rating: 
item_22: Therapist helps the client to make plans and set goals consistent with their values.
Rating: 
Engaged Response Style (Inconsistent)
item_23: Therapist imposes their own, other's or society's values upon the client.
Rating: 
item_24: Therapist encourages action without first exploring the client's psychological experiences.
Rating: 
item_25: Therapist encourages the client's proposed plans even when the client has noticed clear impracticalities.
Rating: 

Output Form: Therapist Empathy Scale (TES)
Purpose: To evaluate a therapy transcript based on the Therapist Empathy Scale (TES). The TES measures therapist empathy based only on verbal content and tone as reflected in the text.
Rating Scale:
1: Not at all
2: A little
3: Infrequently
4: Somewhat
5: Quite a bit
6: Considerably
7: Extensively
TES Items to Rate:
item_1: Concern: The therapist seems engaged, involved, and attentive to what the client has said.
Rating: 
item_2: Expressiveness: The therapist speaks with energy and varies their style to accommodate the mood of the client.
Rating: 
item_3: Resonate or capture client feelings: The therapist's words match the client's emotional state or underscore how the client feels.
Rating: 
item_4: Warmth: The therapist speaks in a friendly, cordial, and sincere manner; seems kindly disposed toward the client.
Rating: 
item_5: Attuned to client's inner world: The therapist provides moment-to-moment verbal acknowledgement of the client's feelings, perceptions, memories, and values.
Rating: 
item_6: Understanding cognitive framework: The therapist clearly follows what the client has said and accurately reflects this understanding; they are on the same page.
Rating: 
item_7: Understanding feelings/inner experience: The therapist shows a sensitive appreciation and gentle caring for the client's emotional state; accurately reflects how the client feels.
Rating: 
item_8: Acceptance of feelings/inner experiences: The therapist validates the client's experience and reflects feelings without judgment or a dismissive attitude.
Rating: 
item_9: Responsiveness: The therapist adjusts their responses to the client's statements and follows the client's lead.
Rating: """

FORMATTING_REMINDER = """
IMPORTANT: You must provide your response in the EXACT format shown above. Each item must be on its own line with "Rating: " followed by a single number. Do not use any markdown formatting, bold text, or asterisks. Just plain text with the exact format shown."""

CLEANUP_PROMPT = """The following is an evaluation response that needs to be cleaned and formatted properly. Extract all the ratings and format them exactly as shown below:

For ACT-FM items (items 1-25), use ratings from 0-3.
For TES items (items 1-9), use ratings from 1-7.

Return ONLY the formatted output with no additional text or explanation:

Output Form: ACT Fidelity Measure (ACT-FM)
item_1: Rating: [number]
item_2: Rating: [number]
...continue for all 25 ACT-FM items...

Output Form: Therapist Empathy Scale (TES)
item_1: Rating: [number]
item_2: Rating: [number]
...continue for all 9 TES items...

Here is the response to clean:
"""

def create_evaluation_prompt(transcript: str, add_formatting_reminder: bool = False) -> str:
    prompt = f"{INSTRUCTION_PROMPT}\n\n<ACT_Transcript>\n{transcript}\n</ACT_Transcript>\n\n<Form>\n{BLANK_FORM_TEMPLATE}\n</Form>"
    if add_formatting_reminder:
        prompt += f"\n\n{FORMATTING_REMINDER}"
    return prompt

def make_api_call_with_backoff(prompt: str, model: str = LLM_JUDGE_MODEL, attempt: int = 1) -> Optional[str]:
    """Make API call with exponential backoff for rate limits"""
    try:
        time.sleep(RATE_LIMIT_DELAY)
        
        chat_response = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return chat_response.choices[0].message.content
        
    except Exception as e:
        error_str = str(e)
        logging.warning(f"API call failed (attempt {attempt}): {e}")
        
        if "429" in error_str or "rate" in error_str.lower():
            backoff_delay = min(BACKOFF_BASE_DELAY * (2 ** (attempt - 1)), MAX_BACKOFF_DELAY)
            logging.info(f"Rate limit hit. Waiting {backoff_delay} seconds before retry...")
            time.sleep(backoff_delay)
            return make_api_call_with_backoff(prompt, model, attempt + 1)
        else:
            time.sleep(BACKOFF_BASE_DELAY)
            return make_api_call_with_backoff(prompt, model, attempt + 1)

def parse_and_validate_ratings(response_text: str) -> dict:
    """Enhanced parser for LLM judge output"""
    report = {
        "act_fm_ratings": {}, "tes_ratings": {},
        "missing_act_fm_items": [], "missing_tes_items": [],
        "parsing_status": "OK", "parsing_warnings_generated": False
    }
    
    if not response_text or not response_text.strip():
        report["parsing_status"] = "API_FAILED_EMPTY_RESPONSE"
        return report

    try:
        cleaned_text = response_text.replace('*', '').replace('#', '').replace('`', '')
        
        patterns = [
            re.compile(r"item_(\d+):.*?Rating:\s*(\d+)", re.DOTALL | re.IGNORECASE),
            re.compile(r"item_(\d+).*?Rating.*?:\s*(\d+)", re.DOTALL | re.IGNORECASE),
            re.compile(r"item_(\d+)[^\d]{1,100}(\d+)", re.DOTALL | re.IGNORECASE)
        ]

        expected_act_fm = set(range(1, 26))
        expected_tes = set(range(1, 10))
        found_act_fm, found_tes = set(), set()
        
        act_section_match = re.search(r"ACT[-\s]*F(?:idelity)?[-\s]*M(?:easure)?|ACT-FM", cleaned_text, re.IGNORECASE)
        tes_section_match = re.search(r"Therapist\s+Empathy\s+Scale|TES", cleaned_text, re.IGNORECASE)
        
        act_text = cleaned_text
        tes_text = cleaned_text
        
        if act_section_match and tes_section_match:
            if act_section_match.start() < tes_section_match.start():
                act_text = cleaned_text[act_section_match.start():tes_section_match.start()]
                tes_text = cleaned_text[tes_section_match.start():]
            else:
                tes_text = cleaned_text[tes_section_match.start():act_section_match.start()]
                act_text = cleaned_text[act_section_match.start():]
        
        for pattern in patterns:
            for match in pattern.findall(act_text):
                try:
                    item_num = int(match[0])
                    rating = int(match[1])
                    
                    if item_num in expected_act_fm and 0 <= rating <= 3:
                        item_key = f"item_{item_num}"
                        if item_key not in report["act_fm_ratings"]:
                            report["act_fm_ratings"][item_key] = rating
                            found_act_fm.add(item_num)
                except (ValueError, IndexError):
                    continue
        
        for pattern in patterns:
            for match in pattern.findall(tes_text):
                try:
                    item_num = int(match[0])
                    rating = int(match[1])
                    
                    if item_num in expected_tes and 1 <= rating <= 7:
                        item_key = f"item_{item_num}"
                        if item_key not in report["tes_ratings"]:
                            report["tes_ratings"][item_key] = rating
                            found_tes.add(item_num)
                except (ValueError, IndexError):
                    continue
        
        missing_act_fm = expected_act_fm - found_act_fm
        missing_tes = expected_tes - found_tes
        
        report["missing_act_fm_items"] = [f"item_{i}" for i in sorted(missing_act_fm)]
        report["missing_tes_items"] = [f"item_{i}" for i in sorted(missing_tes)]
        
        if report["missing_act_fm_items"] or report["missing_tes_items"]:
            if len(found_act_fm) > 0 or len(found_tes) > 0:
                report["parsing_status"] = "PARTIAL"
            else:
                report["parsing_status"] = "FAILED_NO_MATCHES_FOUND"
        
        if 0 < len(found_act_fm) < 25:
            report["parsing_warnings_generated"] = True
        if 0 < len(found_tes) < 9:
            report["parsing_warnings_generated"] = True
            
    except Exception as e:
        logging.error(f"Critical error during parsing: {e}")
        report["parsing_status"] = "FAILED_PARSING_ERROR"
        
    return report

def get_evaluation_with_retry(transcript: str, session_id: int, model_type: str) -> str:
    """Get evaluation with parsing validation and retry logic"""
    attempt = 0
    
    while True:
        attempt += 1
        logging.info(f"Evaluation attempt {attempt} for session {session_id}")
        
        if attempt == 1:
            prompt = create_evaluation_prompt(transcript, add_formatting_reminder=False)
            response = make_api_call_with_backoff(prompt, LLM_JUDGE_MODEL)
        elif attempt == 2:
            logging.info(f"Attempting to clean response with {LLM_CLEANUP_MODEL}")
            cleanup_prompt = CLEANUP_PROMPT + response
            response = make_api_call_with_backoff(cleanup_prompt, LLM_CLEANUP_MODEL)
        else:
            logging.info(f"Retrying with formatting reminder (attempt {attempt})")
            prompt = create_evaluation_prompt(transcript, add_formatting_reminder=True)
            response = make_api_call_with_backoff(prompt, LLM_JUDGE_MODEL)
        
        if not response:
            logging.error(f"Empty response on attempt {attempt}, retrying...")
            continue
        
        parsing_report = parse_and_validate_ratings(response)
        
        if parsing_report["parsing_status"] == "OK":
            logging.info(f"Successfully parsed evaluation for session {session_id} on attempt {attempt}")
            return response
        
        logging.warning(f"Parsing failed on attempt {attempt}: {parsing_report['parsing_status']}")
        logging.warning(f"Missing ACT-FM items: {parsing_report['missing_act_fm_items']}")
        logging.warning(f"Missing TES items: {parsing_report['missing_tes_items']}")

def collect_transcripts(original_dirs: list) -> list:
    """Collect all transcripts with metadata"""
    logging.info("--- Phase 1: Collecting Transcripts ---")
    
    valid_tasks = []
    
    print("\n--- Transcript Analysis Report ---")
    for original_dir in original_dirs:
        if not os.path.isdir(original_dir):
            logging.warning(f"Original input directory not found: '{original_dir}'. Skipping.")
            continue
            
        model_type = os.path.basename(original_dir).replace('simulation_outputs_', '')
        
        for filename in sorted(os.listdir(original_dir)):
            if not filename.endswith("_clean_dialogue.txt"):
                continue

            original_path = os.path.join(original_dir, filename)
            try:
                with open(original_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                turn_count = len(re.findall(r"^(Therapist:|Patient:)", content, re.MULTILINE))
                therapist_turns = len(re.findall(r"^Therapist:", content, re.MULTILINE))
                patient_turns = len(re.findall(r"^Patient:", content, re.MULTILINE))
                word_count = len(content.split())
                char_count = len(content)
                
                print(f"File: {original_path}")
                print(f"  Turns: {turn_count} (T:{therapist_turns}, P:{patient_turns})")
                print(f"  Words: {word_count}, Characters: {char_count}\n")

                session_id_match = re.search(r'simulation_(\d+)_', filename)
                if session_id_match:
                    session_id = int(session_id_match.group(1))
                    valid_tasks.append({
                        'session_id': session_id,
                        'model_type': model_type,
                        'transcript_path': original_path,
                        'turn_count': turn_count,
                        'therapist_turns': therapist_turns,
                        'patient_turns': patient_turns,
                        'word_count': word_count,
                        'char_count': char_count
                    })

            except Exception as e:
                logging.error(f"Error processing {original_path}: {e}")

    logging.info(f"--- Collection complete. Found {len(valid_tasks)} transcripts. ---")
    return valid_tasks

def calculate_act_fm_scores(ratings: dict) -> dict:
    if not ratings:
        return {}
    
    r = {f'item_{i}': ratings.get(f'item_{i}', 0) for i in range(1, 26)}
    scores = {}
    
    c_stance_raw = r['item_1'] + r['item_2'] + r['item_3'] + r['item_4']
    scores['act_fm_consistent_stance'] = (c_stance_raw / 4) * 3
    scores['act_fm_inconsistent_stance'] = r['item_5'] + r['item_6'] + r['item_7']
    scores['act_fm_consistent_open'] = r['item_8'] + r['item_9'] + r['item_10']
    scores['act_fm_inconsistent_open'] = r['item_11'] + r['item_12'] + r['item_13']
    scores['act_fm_consistent_aware'] = r['item_14'] + r['item_15'] + r['item_16']
    scores['act_fm_inconsistent_aware'] = r['item_17'] + r['item_18'] + r['item_19']
    scores['act_fm_consistent_engaged'] = r['item_20'] + r['item_21'] + r['item_22']
    scores['act_fm_inconsistent_engaged'] = r['item_23'] + r['item_24'] + r['item_25']
    scores['act_fm_total_consistency'] = (scores['act_fm_consistent_stance'] + scores['act_fm_consistent_open'] + 
                                         scores['act_fm_consistent_aware'] + scores['act_fm_consistent_engaged'])
    scores['act_fm_total_inconsistency'] = (scores['act_fm_inconsistent_stance'] + scores['act_fm_inconsistent_open'] + 
                                           scores['act_fm_inconsistent_aware'] + scores['act_fm_inconsistent_engaged'])
    return scores

def calculate_tes_score(ratings: dict) -> dict:
    if not ratings:
        return {}
    item_scores = list(ratings.values())
    return {'tes_total_empathy_mean': sum(item_scores) / len(item_scores) if item_scores else 0}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FULL_EVALS_DIR, exist_ok=True)
    
    if os.path.exists(PARSING_ERROR_LOG_FILE):
        os.remove(PARSING_ERROR_LOG_FILE)
    
    error_logger = logging.getLogger('parsing_errors')
    error_logger.setLevel(logging.INFO)
    error_logger.propagate = False
    handler = logging.FileHandler(PARSING_ERROR_LOG_FILE)
    handler.setFormatter(logging.Formatter('%(message)s'))
    error_logger.addHandler(handler)

    all_tasks = collect_transcripts(ORIGINAL_INPUT_DIRS)
    
    if not all_tasks:
        logging.error("No valid files found for evaluation. Exiting.")
        return

    logging.info(f"\n--- Phase 2: Sequential Evaluation Generation ---")
    
    total_tasks = len(all_tasks)
    completed = 0
    
    for task in all_tasks:
        completed += 1
        session_id = task['session_id']
        model_type = task['model_type']
        
        logging.info(f"\n[{completed}/{total_tasks}] Processing session {session_id} ({model_type})")
        
        eval_output_dir = os.path.join(FULL_EVALS_DIR, model_type)
        os.makedirs(eval_output_dir, exist_ok=True)
        eval_file_path = os.path.join(eval_output_dir, f"eval_session_{session_id}.txt")
        task['eval_file_path'] = eval_file_path

        if os.path.exists(eval_file_path) and os.path.getsize(eval_file_path) > 0:
            logging.info(f"Evaluation already exists, skipping.")
            continue
        
        with open(task['transcript_path'], 'r', encoding='utf-8') as f:
            transcript_text = f.read()
        
        response_text = get_evaluation_with_retry(transcript_text, session_id, model_type)
        
        with open(eval_file_path, 'w', encoding='utf-8') as f:
            f.write(response_text)
        
        logging.info(f"Successfully saved evaluation for session {session_id}")
        
        if completed % 10 == 0:
            remaining = total_tasks - completed
            logging.info(f"Overall progress: {completed}/{total_tasks} completed. Remaining: {remaining}")

    logging.info(f"\n--- Phase 3: Parsing and Aggregating Results ---")
    all_results = []
    
    for task in all_tasks:
        session_id = task['session_id']
        model_type = task['model_type']
        eval_file_path = task['eval_file_path']

        result_row = {
            'session_id': session_id,
            'model_type': model_type,
            'turn_count': task.get('turn_count', 0),
            'therapist_turns': task.get('therapist_turns', 0),
            'patient_turns': task.get('patient_turns', 0),
            'word_count': task.get('word_count', 0),
            'char_count': task.get('char_count', 0)
        }

        if not os.path.exists(eval_file_path):
            logging.warning(f"Evaluation file not found for session {session_id}.")
            result_row.update({"parsing_status": "API_FAILED_NO_FILE"})
            all_results.append(result_row)
            continue
        
        with open(eval_file_path, 'r', encoding='utf-8') as f:
            response_text = f.read()
        
        parsing_report = parse_and_validate_ratings(response_text)
        
        if parsing_report["parsing_status"] != "OK":
            log_msg = f"--- PARSING ISSUE: session_{session_id} ({model_type}) ---\n"
            log_msg += f"Status: {parsing_report['parsing_status']}\n"
            if parsing_report["missing_act_fm_items"]:
                log_msg += f"Missing ACT-FM: {parsing_report['missing_act_fm_items']}\n"
            if parsing_report["missing_tes_items"]:
                log_msg += f"Missing TES: {parsing_report['missing_tes_items']}\n"
            log_msg += "--- RAW TEXT ---\n"
            log_msg += response_text + "\n"
            log_msg += "--------------------------------------------------------\n"
            error_logger.info(log_msg)

        for item_key, rating in parsing_report['act_fm_ratings'].items():
            result_row[f'act_fm_{item_key}'] = rating
        
        for item_key, rating in parsing_report['tes_ratings'].items():
            result_row[f'tes_{item_key}'] = rating
        
        act_fm_scores = calculate_act_fm_scores(parsing_report['act_fm_ratings'])
        tes_scores = calculate_tes_score(parsing_report['tes_ratings'])
        
        result_row.update(act_fm_scores)
        result_row.update(tes_scores)
        
        result_row['parsing_status'] = parsing_report['parsing_status']
        result_row['num_missing_act_fm'] = len(parsing_report['missing_act_fm_items'])
        result_row['num_missing_tes'] = len(parsing_report['missing_tes_items'])
        
        all_results.append(result_row)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        id_cols = ['session_id', 'model_type', 'turn_count', 'therapist_turns', 'patient_turns', 'word_count', 'char_count']
        parsing_cols = ['parsing_status', 'num_missing_act_fm', 'num_missing_tes']
        score_cols = [col for col in results_df.columns if ('act_fm_' in col or 'tes_' in col) and not col.startswith('act_fm_item') and not col.startswith('tes_item')]
        item_cols = [col for col in results_df.columns if col.startswith('act_fm_item') or col.startswith('tes_item')]
        
        existing_cols = [col for col in id_cols + parsing_cols + sorted(score_cols) + sorted(item_cols) if col in results_df.columns]
        results_df = results_df[existing_cols]
        
        results_df.to_csv(OUTPUT_RESULTS_CSV, index=False)
        logging.info(f"Successfully saved all aggregated results to {OUTPUT_RESULTS_CSV}")
        
        parsing_summary = results_df['parsing_status'].value_counts()
        logging.info(f"\nParsing Summary:\n{parsing_summary}")
        
        total_evals = len(results_df)
        successful_evals = len(results_df[results_df['parsing_status'] == 'OK'])
        logging.info(f"\nSuccessfully parsed: {successful_evals}/{total_evals} evaluations")
    else:
        logging.warning("No results were generated to save.")

if __name__ == '__main__':
    main()