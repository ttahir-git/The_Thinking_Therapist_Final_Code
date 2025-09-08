import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import json
import logging
import warnings
import shutil
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error

# --- Configuration and Constants ---

BASE_EVAL_DIR = "final_evaluations_mistral_sequential_Aug_19"
PRE_EXISTING_CSV_PATH = os.path.join(BASE_EVAL_DIR, 'evaluation_results_with_parsing_health.csv')
RAW_EVALS_DIR = os.path.join(BASE_EVAL_DIR, 'full_text_evaluations')
RAW_TRANSCRIPT_DIRS = [
    'simulation_outputs_llama3_3b_instruct', 'simulation_outputs_orpo',
    'simulation_outputs_orpo_no_COT', 'simulation_outputs_sft',
    'simulation_outputs_sft_answer_only', 'simulation_outputs_llama3_3b_instruct_no_COT'
]
PATIENT_PROFILES_FILE = 'standardized_patient_profiles.json'

OUTPUT_DIR = "unified_analysis_outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
PARSED_EVALS_CSV = os.path.join(OUTPUT_DIR, 'parsed_evaluation_scores.csv')
EXCLUSION_REPORT_FILE = os.path.join(OUTPUT_DIR, "exclusion_report.txt")
SUBGROUP_REPORT_FILE = os.path.join(OUTPUT_DIR, 'subgroup_analysis_summary_report.txt')
JSON_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'llm_friendly_analysis_summary.json') # LLM-friendly JSON output file

DEPENDENT_VARIABLES = ['act_fm_total', 'tes_mean']
ALPHA = 0.05

# --- Setup ---

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# Global dictionary to hold all analysis results for JSON output
analysis_results_json = {
    "study_metadata": {
        "title": "The Thinking Therapist: Training Large Language Models to Deliver Acceptance and Commitment Therapy Informed Interactions using Supervised Learning and Reinforcement Learning Approaches",
        "description": "This JSON file contains a structured summary of the statistical analyses performed in the study. It is designed to be easily parsed by large language models.",
        "primary_outcome_measures": [
            {"name": "ACT-FM Total Consistency Score", "variable_name": "act_fm_total", "description": "The sum of the four consistency subscales of the ACT Fidelity Measure (consistent stance, open, aware, and engaged). Higher scores indicate greater adherence to ACT principles."},
            {"name": "TES Total Empathy Mean", "variable_name": "tes_mean", "description": "The average of all nine Therapist Empathy Scale items. Higher scores indicate greater therapist empathy."}
        ]
    },
    "data_summary": {},
    "descriptive_statistics": {},
    "primary_analysis": {
        "linear_mixed_effects_models": {},
        "correlation_analysis": {}
    },
    "exploratory_subgroup_analysis": {
        "head_to_head_win_rates": {},
        "feature_importance": {},
        "interaction_analysis": {},
        "cot_impact_analysis": {},
        "optimal_model_by_subgroup": {}
    }
}

# --- Part 1: Data Parsing and Preparation Functions ---

def parse_and_validate_ratings(response_text: str) -> dict:
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
            re.compile(r"(?:\**)?(item_\d+)(?:\**)?:.*?Rating:\s*(?:\**)?\s*(\d+)\s*(?:\**)?", re.DOTALL | re.IGNORECASE),
            re.compile(r"\*\*(item_\d+)\*\*:.*?Rating:\s*\*\*(\d+)\*\*", re.DOTALL | re.IGNORECASE),
            re.compile(r"(item_\d+).*?\n.*?Rating.*?(\d+)", re.DOTALL | re.IGNORECASE),
            re.compile(r"(item_\d+).*?Rating\s*:\s*(\d+)", re.DOTALL | re.IGNORECASE),
            re.compile(r"(item_\d+)[^\d]{1,100}(\d+)", re.DOTALL | re.IGNORECASE)
        ]

        expected_act_fm = {f"item_{i}" for i in range(1, 26)}
        expected_tes = {f"item_{i}" for i in range(1, 10)}
        found_act_fm, found_tes = set(), set()

        act_key_phrases = ["ACT Fidelity Measure", "ACT-FM", "Output Form: ACT", "ACT Fidelity", "Acceptance and Commitment Therapy"]
        tes_key_phrases = ["Therapist Empathy Scale", "TES", "Output Form: Therapist", "Empathy Scale", "TES Items"]

        act_match = None
        tes_match = None

        for phrase in act_key_phrases:
            match = re.search(phrase, response_text, re.IGNORECASE)
            if match and (act_match is None or match.start() < act_match.start()):
                act_match = match

        for phrase in tes_key_phrases:
            match = re.search(phrase, response_text, re.IGNORECASE)
            if match and (tes_match is None or match.start() < tes_match.start()):
                tes_match = match

        act_text, tes_text = "", ""
        if act_match and tes_match:
            if act_match.start() < tes_match.start():
                act_text = response_text[act_match.start():tes_match.start()]
                tes_text = response_text[tes_match.start():]
            else:
                tes_text = response_text[tes_match.start():act_match.start()]
                act_text = response_text[act_match.start():]
        elif act_match:
            act_text = response_text[act_match.start():]
        elif tes_match:
            tes_text = response_text[tes_match.start():]
        else:
            act_text = response_text
            tes_text = response_text

        if act_text:
            for pattern in patterns:
                matches = pattern.findall(act_text)
                for match in matches:
                    if len(match) >= 2:
                        item_key = match[0].lower().replace('*', '').strip()
                        rating_str = match[1].replace('*', '').strip()

                        item_key = re.sub(r'\s+', '_', item_key)
                        if not item_key.startswith('item_'):
                            continue

                        try:
                            rating = int(rating_str)
                            if item_key in expected_act_fm and 0 <= rating <= 3:
                                if item_key not in report["act_fm_ratings"]:
                                    report["act_fm_ratings"][item_key] = rating
                                    found_act_fm.add(item_key)
                        except ValueError:
                            continue

        if tes_text:
            for pattern in patterns:
                matches = pattern.findall(tes_text)
                for match in matches:
                    if len(match) >= 2:
                        item_key = match[0].lower().replace('*', '').strip()
                        rating_str = match[1].replace('*', '').strip()

                        item_key = re.sub(r'\s+', '_', item_key)
                        if not item_key.startswith('item_'):
                            continue

                        try:
                            rating = int(rating_str)
                            if item_key in expected_tes and 1 <= rating <= 7:
                                if item_key not in report["tes_ratings"]:
                                    report["tes_ratings"][item_key] = rating
                                    found_tes.add(item_key)
                        except ValueError:
                            continue

        if len(found_act_fm) < 10 or len(found_tes) < 5:
            for pattern in patterns:
                matches = pattern.findall(cleaned_text)
                for match in matches:
                    if len(match) >= 2:
                        item_key = match[0].lower().replace('*', '').strip()
                        rating_str = match[1].replace('*', '').strip()

                        item_key = re.sub(r'\s+', '_', item_key)
                        if not item_key.startswith('item_'):
                            continue

                        try:
                            rating = int(rating_str)

                            if item_key in expected_act_fm and 0 <= rating <= 3:
                                if item_key not in report["act_fm_ratings"]:
                                    report["act_fm_ratings"][item_key] = rating
                                    found_act_fm.add(item_key)
                            elif item_key in expected_tes and 1 <= rating <= 7:
                                if item_key not in report["tes_ratings"]:
                                    report["tes_ratings"][item_key] = rating
                                    found_tes.add(item_key)
                        except ValueError:
                            continue

        report["missing_act_fm_items"] = sorted(list(expected_act_fm - found_act_fm))
        report["missing_tes_items"] = sorted(list(expected_tes - found_tes))

        if report["missing_act_fm_items"] or report["missing_tes_items"]:
            if len(found_act_fm) > 0 or len(found_tes) > 0:
                report["parsing_status"] = "PARTIAL"
            else:
                report["parsing_status"] = "FAILED_NO_MATCHES_FOUND"

        if len(found_act_fm) > 0 and len(found_act_fm) < 20:
            report["parsing_warnings_generated"] = True
        if len(found_tes) > 0 and len(found_tes) < 7:
            report["parsing_warnings_generated"] = True

    except Exception as e:
        logging.error(f"Critical error during parsing: {e}")
        report["parsing_status"] = "FAILED_PARSING_ERROR"

    return report

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

    scores['act_fm_total_consistency'] = (scores['act_fm_consistent_stance'] +
                                         scores['act_fm_consistent_open'] +
                                         scores['act_fm_consistent_aware'] +
                                         scores['act_fm_consistent_engaged'])
    scores['act_fm_total_inconsistency'] = (scores['act_fm_inconsistent_stance'] +
                                           scores['act_fm_inconsistent_open'] +
                                           scores['act_fm_inconsistent_aware'] +
                                           scores['act_fm_inconsistent_engaged'])

    return scores

def calculate_tes_score(ratings: dict) -> dict:
    if not ratings:
        return {}

    item_scores = list(ratings.values())
    if not item_scores:
        return {'tes_total_empathy_mean': 0}

    return {'tes_total_empathy_mean': sum(item_scores) / len(item_scores)}

def generate_csv_from_evaluations():
    logging.info("--- Generating CSV from Raw Evaluation Files ---")

    if not os.path.exists(RAW_EVALS_DIR):
        logging.error(f"Evaluation directory '{RAW_EVALS_DIR}' not found. Cannot generate CSV.")
        return False

    all_results = []

    for model_type in os.listdir(RAW_EVALS_DIR):
        model_dir = os.path.join(RAW_EVALS_DIR, model_type)
        if not os.path.isdir(model_dir):
            continue

        logging.info(f"Processing model type: {model_type}")

        for filename in os.listdir(model_dir):
            if not filename.startswith('eval_session_') or not filename.endswith('.txt'):
                continue

            session_match = re.search(r'eval_session_(\d+)\.txt', filename)
            if not session_match:
                continue

            session_id = int(session_match.group(1))
            filepath = os.path.join(model_dir, filename)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    evaluation_text = f.read()

                if not evaluation_text.strip():
                    logging.warning(f"  Warning: Empty evaluation file for session {session_id}")
                    continue

                parse_result = parse_and_validate_ratings(evaluation_text)

                act_fm_scores = calculate_act_fm_scores(parse_result['act_fm_ratings'])
                tes_scores = calculate_tes_score(parse_result['tes_ratings'])

                result = {
                    'session_id': session_id,
                    'model_type': model_type,
                    'parsing_status': parse_result['parsing_status'],
                    'parsing_warnings_generated': parse_result['parsing_warnings_generated'],
                    'missing_act_fm_items': ','.join(parse_result['missing_act_fm_items']),
                    'missing_tes_items': ','.join(parse_result['missing_tes_items'])
                }

                for i in range(1, 26):
                    item_key = f'item_{i}'
                    result[f'act_fm_{item_key}'] = parse_result['act_fm_ratings'].get(item_key, np.nan)

                for i in range(1, 10):
                    item_key = f'item_{i}'
                    result[f'tes_{item_key}'] = parse_result['tes_ratings'].get(item_key, np.nan)

                result.update(act_fm_scores)
                result.update(tes_scores)

                all_results.append(result)

            except Exception as e:
                logging.error(f"  Error processing {filepath}: {e}")
                continue

    if not all_results:
        logging.error("No evaluation results could be processed.")
        return False

    df = pd.DataFrame(all_results)
    df.to_csv(PARSED_EVALS_CSV, index=False)
    logging.info(f"Successfully generated CSV with {len(df)} records: '{PARSED_EVALS_CSV}'")
    logging.info(f"Model types found: {sorted(df['model_type'].unique())}")
    logging.info(f"Parsing status distribution:\n{df['parsing_status'].value_counts()}")

    return True

def load_and_merge_data():
    logging.info("--- Loading and Merging Data ---")

    try:
        eval_df = pd.read_csv(PARSED_EVALS_CSV)
    except FileNotFoundError:
        logging.error(f"CRITICAL: Parsed evaluations CSV not found at '{PARSED_EVALS_CSV}'. Exiting.")
        return None

    try:
        with open(PATIENT_PROFILES_FILE, 'r', encoding='utf-8') as f:
            patient_profiles = json.load(f)
    except FileNotFoundError:
        logging.error(f"CRITICAL: Patient profiles file not found at '{PATIENT_PROFILES_FILE}'. Exiting.")
        return None
    patient_df = pd.DataFrame(patient_profiles)
    patient_df['session_id'] = patient_df.index + 1

    merged_df = pd.merge(patient_df, eval_df, on='session_id', how='inner')

    if merged_df.empty:
        logging.error("Merging resulted in an empty DataFrame. Check 'session_id' alignment.")
        return None

    logging.info(f"Successfully processed {len(merged_df)} simulation results.")
    return merged_df

def process_and_engineer_features(df):
    logging.info("--- Processing Data and Engineering Features ---")

    df.rename(columns={
        'act_fm_total_consistency': 'act_fm_total',
        'tes_total_empathy_mean': 'tes_mean',
    }, inplace=True)

    if 'model' in df.columns and 'model_type' not in df.columns:
        df.rename(columns={'model': 'model_type'}, inplace=True)

    df.rename(columns={'model_type': 'model_id'}, inplace=True)

    model_name_mapping = {
        'llama3_3b_instruct': 'Instruct (COT)',
        'llama3_3b_instruct_no_COT': 'Instruct (no COT)',
        'sft': 'SFT (COT)',
        'sft_answer_only': 'SFT (no COT)',
        'orpo': 'ORPO (COT)',
        'orpo_no_COT': 'ORPO (no COT)'
    }
    df['model'] = df['model_id'].map(model_name_mapping).fillna(df['model_id'])

    # Define keyword lists for parsing new features
    occupations = ["software developer", "teacher", "nurse", "artist", "accountant", "student", "manager", "construction worker", "chef", "social worker", "business owner", "unemployed", "data scientist"]
    mh_issues = ["PTSD", "social anxiety", "OCD", "burnout", "adjustment disorder", "low self-esteem", "grief", "mild anxiety", "moderate depression", "generalized anxiety disorder"]
    life_events = ["a recent difficult breakup", "the loss of a loved one", "job loss or instability", "a recent move", "ongoing financial stress", "starting a demanding new job or school program", "significant family conflict", "a health scare", "feeling isolated or lonely", "major life transition"]
    coping_mechanisms = ["talking to friends/family", "avoiding triggers", "engaging in hobbies", "exercise", "mindfulness/meditation", "overworking", "substance use (mild/moderate)", "seeking reassurance", "intellectualizing feelings", "emotional eating", "procrastination", "using humor/sarcasm"]
    relationship_statuses = ["single", "in a relationship", "married", "divorced", "widowed"]
    support_systems = ["a few close friends", "a supportive partner", "limited social support currently", "supportive family", "relies mostly on self", "colleagues provide some support"]

    def get_feature(text, keywords):
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                # Special handling for anxiety to avoid misclassifying social anxiety as mild anxiety
                if keyword == "mild anxiety" and "social anxiety" in text_lower:
                    continue
                return keyword
        return "unknown"

    # --- Feature Extraction from full_scenario_text ---
    df['age_lower'] = df['full_scenario_text'].apply(lambda x: int(re.search(r'\((\d+)-\d+\)', x).group(1)) if re.search(r'\((\d+)-\d+\)', x) else np.nan)
    df['gender'] = df['full_scenario_text'].apply(lambda x: 'male' if 'identifies as male' in x else ('female' if 'identifies as female' in x else 'non-binary'))
    df['occupation'] = df['full_scenario_text'].apply(lambda x: get_feature(x, occupations))
    df['primary_issue'] = df['full_scenario_text'].apply(lambda x: get_feature(x, mh_issues))
    
    # NEW: Extract additional patient profile components
    df['life_event'] = df['full_scenario_text'].apply(lambda x: get_feature(x, life_events))
    df['coping_mechanism'] = df['full_scenario_text'].apply(lambda x: get_feature(x, coping_mechanisms))
    df['relationship_status'] = df['full_scenario_text'].apply(lambda x: get_feature(x, relationship_statuses))
    df['support_system'] = df['full_scenario_text'].apply(lambda x: get_feature(x, support_systems))

    # --- Feature Creation ---
    df['has_cot'] = ~df['model'].str.contains('no COT', regex=False)
    df['model_family'] = df['model'].apply(lambda x: x.split(' ')[0])

    logging.info("Feature engineering complete. Added life_event, coping_mechanism, relationship_status, support_system.")
    return df


# --- Part 2: Reporting and Helper Functions ---

def generate_exclusion_report(df_raw, original_dirs, output_path):
    logging.info("--- Generating Exclusion Report ---")

    all_original_files = set()
    for original_dir in original_dirs:
        full_dir_path = os.path.join(os.path.dirname(BASE_EVAL_DIR), original_dir)
        if not os.path.isdir(full_dir_path):
            logging.warning(f"Original source directory not found: '{full_dir_path}'. Skipping for report.")
            continue
        model_type = os.path.basename(original_dir).replace('simulation_outputs_', '')
        for filename in os.listdir(full_dir_path):
            match = re.search(r'simulation_(\d+)_', filename)
            if match:
                session_id = int(match.group(1))
                all_original_files.add((model_type, session_id, filename))

    files_in_csv = set(tuple(x) for x in df_raw[['model_id', 'session_id']].to_numpy())
    original_files_by_id = {(model, sid): fname for model, sid, fname in all_original_files}
    csv_ids = {(model, sid) for model, sid in files_in_csv}
    files_dropped_pre_eval = original_files_by_id.keys() - csv_ids

    excluded_files = []
    for model_type, session_id in sorted(list(files_dropped_pre_eval)):
        filename = original_files_by_id.get((model_type, session_id), "N/A")
        excluded_files.append({
            'model': model_type, 'session_id': session_id, 'filename': filename,
            'reason': "File not found in evaluation CSV. Likely failed pre-evaluation filter."})

    valid_statuses = ['OK', 'PARTIAL']
    for _, row in df_raw.iterrows():
        if (row['model_id'], row['session_id']) in files_dropped_pre_eval:
            continue
        reason = ""
        if row['parsing_status'] not in valid_statuses:
            reason = f"Invalid parsing status: '{row['parsing_status']}'."
        elif pd.isna(row.get('act_fm_total', np.nan)) or pd.isna(row.get('tes_mean', np.nan)):
            missing = [v for v in ['act_fm_total', 'tes_mean'] if pd.isna(row.get(v, np.nan))]
            reason = f"Missing key data for analysis: {', '.join(missing)}."
        if reason:
            filename = original_files_by_id.get((row['model_id'], row['session_id']), "N/A")
            excluded_files.append({
                'model': row['model_id'], 'session_id': row['session_id'], 'filename': filename, 'reason': reason})

    if not excluded_files:
        logging.info("No files were excluded from the analysis.")
        return

    excluded_files.sort(key=lambda x: (x['model'], x['session_id']))
    with open(output_path, 'w') as f:
        f.write("="*80 + "\nEXCLUSION REPORT\n" + "="*80 + "\n")
        current_model = ""
        for item in excluded_files:
            if item['model'] != current_model:
                current_model = item['model']
                f.write(f"\n--- Model Type: {current_model} ---\n")
            f.write(f"  - File: {item['filename']} (Session ID: {item['session_id']})\n")
            f.write(f"    Reason: {item['reason']}\n")
    logging.info(f"Exclusion report with {len(excluded_files)} entries saved to '{output_path}'")

def rank_biserial_correlation(U, n1, n2):
    if n1 == 0 or n2 == 0:
        return 0.0
    return 1 - (2 * U) / (n1 * n2)

def create_publication_tables(df, test_results=None):
    logging.info("--- Creating Publication-Ready Tables ---")

    table1_data = []
    for model in sorted(df['model'].unique()):
        model_data = df[df['model'] == model]
        for var_name, display_name in [('act_fm_total', 'ACT-FM Consistency'),
                                        ('tes_mean', 'TES Empathy')]:
            data = model_data[var_name].dropna()
            if len(data) > 0:
                table1_data.append({
                    'Model': model, 'Measure': display_name, 'n': len(data),
                    'M (SD)': f"{data.mean():.2f} ({data.std():.2f})",
                    'Range': f"{data.min():.2f}-{data.max():.2f}"})

    table1_df = pd.DataFrame(table1_data)
    table1_file = os.path.join(TABLES_DIR, 'table1_descriptive_stats.csv')
    table1_df.to_csv(table1_file, index=False)
    logging.info(f"Saved publication Table 1 to '{table1_file}'")

    if test_results is not None:
        table2_file = os.path.join(TABLES_DIR, 'table2_test_results.csv')
        test_results.to_csv(table2_file, index=False)
        logging.info(f"Saved publication Table 2 to '{table2_file}'")


# --- Part 3A: Standard Visualizations ---

def create_primary_visualizations(df):
    logging.info("--- Creating Primary Analysis Visualizations ---")
    model_order = sorted(df['model'].unique())

    for dv in DEPENDENT_VARIABLES:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='model', y=dv, data=df, order=model_order, palette='Set2')
        sns.stripplot(x='model', y=dv, data=df, order=model_order, color='0.25', size=3)
        plt.title(f'Distribution of {dv.replace("_", " ").title()}', fontsize=14)
        plt.xlabel('Model Type', fontsize=12)
        plt.ylabel(dv.replace("_", " ").title(), fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plot_filename = os.path.join(PLOTS_DIR, f'primary_boxplot_{dv}.png')
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved boxplot to '{plot_filename}'")

    corr_subscales = ['act_fm_consistent_stance', 'act_fm_inconsistent_stance',
                      'act_fm_consistent_open', 'act_fm_inconsistent_open',
                      'act_fm_consistent_aware', 'act_fm_inconsistent_aware',
                      'act_fm_consistent_engaged', 'act_fm_inconsistent_engaged', 'tes_mean']
    corr_data = df[corr_subscales].dropna()
    if not corr_data.empty:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = corr_data.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        labels = [s.replace('act_fm_', '').replace('_', ' ').title() for s in corr_subscales]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)
        plt.title('Correlation Matrix of ACT-FM Subscales and TES', fontsize=14, pad=20)
        plt.tight_layout()
        corr_file = os.path.join(PLOTS_DIR, 'primary_correlation_matrix.png')
        plt.savefig(corr_file, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved correlation matrix to '{corr_file}'")
        
        # Save correlation matrix to global JSON object
        analysis_results_json["primary_analysis"]["correlation_analysis"] = {
            "description": "Pearson correlation coefficients between ACT-FM subscales and the total TES empathy score.",
            "matrix": corr_matrix.to_dict()
        }

# --- Part 3B: Additional Publication Visualizations ---

def create_forest_plot_from_mixedlm(results_df, dv, output_dir):
    logging.info(f"--- Creating Forest Plot for {dv} ---")

    plot_data = results_df[results_df['Test'].str.contains(dv)].copy()
    if plot_data.empty:
        logging.warning(f"No data available to create forest plot for {dv}.")
        return

    plot_data['label'] = plot_data['Level A'] + ' vs. ' + plot_data['Level B']
    plot_data.sort_values('Statistic', ascending=True, inplace=True)

    y_pos = np.arange(len(plot_data))

    fig, ax = plt.subplots(figsize=(10, 8))

    errors = [plot_data['Statistic'] - plot_data['ci_lower_95'], plot_data['ci_upper_95'] - plot_data['Statistic']]
    ax.errorbar(x=plot_data['Statistic'], y=y_pos, xerr=errors, fmt='o', color='black',
                ecolor='gray', elinewidth=1, capsize=3, label='Mean Difference & 95% CI')

    ax.axvline(x=0, linestyle='--', color='red', lw=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data['label'])
    ax.set_xlabel('Mean Difference')
    ax.set_ylabel('Pairwise Comparison')
    ax.set_title(f'Pairwise Model Comparisons for {dv.replace("_", " ").title()}')
    plt.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plot_filename = os.path.join(output_dir, f'primary_forest_plot_{dv}.png')
    plt.savefig(plot_filename)
    plt.close()
    logging.info(f"Saved forest plot to '{plot_filename}'")

def create_feature_importance_plot(importances_df, metric_name, output_dir, top_n=15):
    logging.info(f"--- Creating Feature Importance Plot for {metric_name} ---")
    top_features = importances_df.head(top_n).sort_values('importance', ascending=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
    plt.title(f'Top {top_n} Features Predicting {metric_name.replace("_", " ").title()}')
    plt.xlabel('Permutation Importance (mean)')
    plt.ylabel('Feature')
    plt.tight_layout()

    plot_filename = os.path.join(output_dir, f'exploratory_feature_importance_{metric_name}.png')
    plt.savefig(plot_filename)
    plt.close()
    logging.info(f"Saved feature importance plot to '{plot_filename}'")


# --- Part 4: Exploratory Subgroup Analysis Functions ---

def analyze_model_differences_by_characteristic(df, characteristic, metric='act_fm_total'):
    results = {}
    for value in df[characteristic].dropna().unique():
        subset = df[df[characteristic] == value]
        model_scores = subset.groupby('model')[metric].agg(['mean', 'std', 'count'])
        if len(model_scores) < 2: continue

        comparisons = []
        model_pairs = list(combinations(model_scores.index, 2))
        for model1, model2 in model_pairs:
            # Create a temporary DataFrame that merges the scores on session_id
            df_m1 = subset[subset['model'] == model1][['session_id', metric]].rename(columns={metric: 'score_model1'})
            df_m2 = subset[subset['model'] == model2][['session_id', metric]].rename(columns={metric: 'score_model2'})
            paired_data = pd.merge(df_m1, df_m2, on='session_id').dropna()

            if len(paired_data) < 3: continue

            # Use Wilcoxon signed-rank test for paired samples
            stat, p_val = stats.wilcoxon(paired_data['score_model1'], paired_data['score_model2'])
            
            # Append results without the incorrect effect size
            comparisons.append({'model1': model1, 'model2': model2, 
                                'mean_diff': paired_data['score_model1'].mean() - paired_data['score_model2'].mean(),
                                'p_value': p_val})

        if comparisons:
            valid_comps = [c for c in comparisons if c['p_value'] is not None]
            if valid_comps:
                p_values = [c['p_value'] for c in valid_comps]
                reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
                for i, comp in enumerate(valid_comps):
                    comp['p_adj'], comp['significant_adj'] = p_adj[i], reject[i]
                results[value] = {'model_scores': model_scores, 'comparisons': valid_comps}
    return results

def create_subgroup_visualizations(df):
    logging.info("--- Creating Subgroup Analysis Visualizations ---")

    # Expanded list of characteristics to analyze
    characteristics = [
        'archetype_name', 'psych_mindedness_level', 'interaction_style_name', 
        'primary_issue', 'life_event', 'relationship_status', 'support_system'
    ]
    
    for char in characteristics:
        if char not in df.columns or df[char].nunique() < 2: continue
        pivot_data = df.pivot_table(values='act_fm_total', index=char, columns='model', aggfunc='mean')
        if pivot_data.empty: continue

        plt.figure(figsize=(12, max(6, 0.5 * len(pivot_data.index)))) # Dynamic height
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', center=pivot_data.mean().mean())
        plt.title(f'Mean ACT-FM Score by {char.replace("_", " ").title()} and Model')
        plt.xlabel('Model'); plt.ylabel(char.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'subgroup_heatmap_{char}.png'))
        plt.close()

    # Create a grid of bar plots for performance profiles
    num_chars = len(characteristics)
    # Adjusting subplot grid to fit more charts
    n_cols = 2
    n_rows = (num_chars + n_cols - 1) // n_cols 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes_flat = axes.flatten()

    for idx, char in enumerate(characteristics):
        if char not in df.columns or df[char].nunique() < 2: continue
        ax = axes_flat[idx]
        pivot_data = df.groupby([char, 'model'])['act_fm_total'].mean().unstack()
        pivot_data.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'ACT-FM Performance by {char.replace("_", " ").title()}')
        ax.set_xlabel(''); ax.set_ylabel('ACT-FM Total Score')
        ax.legend(title='Model', fontsize=8)
        # FIX: Use plt.setp to correctly apply rotation and alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Hide any unused subplots
    for i in range(num_chars, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'subgroup_performance_profiles.png'))
    plt.close()

    for dv in DEPENDENT_VARIABLES:
        plt.figure(figsize=(12, 7))
        sns.pointplot(data=df, x='model', y=dv, hue='archetype_name', errorbar='ci', dodge=True, capsize=.05)
        plt.title(f'Interaction of Model and Patient Archetype on {dv.replace("_", " ").title()}', fontsize=14)
        plt.xlabel('Model Type', fontsize=12)
        plt.ylabel(dv.replace("_", " ").title(), fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Patient Archetype', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_filename = os.path.join(PLOTS_DIR, f'exploratory_interaction_plot_{dv}.png')
        plt.savefig(plot_filename)
        plt.close()
        logging.info(f"Saved interaction plot to '{plot_filename}'")


def create_win_rate_matrix(df):
    logging.info("--- Calculating Head-to-Head Win Rates ---")
    models = sorted(df['model'].unique())
    win_matrix_act = pd.DataFrame(index=models, columns=models, data=0)
    win_matrix_tes = pd.DataFrame(index=models, columns=models, data=0)
    tie_matrix_act = pd.DataFrame(index=models, columns=models, data=0)
    tie_matrix_tes = pd.DataFrame(index=models, columns=models, data=0)

    for session_id in df['session_id'].unique():
        session_data = df[df['session_id'] == session_id].set_index('model')
        for m1, m2 in combinations(models, 2):
            if m1 in session_data.index and m2 in session_data.index:
                score1_act, score2_act = session_data.loc[m1, 'act_fm_total'], session_data.loc[m2, 'act_fm_total']
                score1_tes, score2_tes = session_data.loc[m1, 'tes_mean'], session_data.loc[m2, 'tes_mean']
                if not pd.isna(score1_act) and not pd.isna(score2_act):
                    if score1_act > score2_act: win_matrix_act.loc[m1, m2] += 1
                    elif score2_act > score1_act: win_matrix_act.loc[m2, m1] += 1
                    else: tie_matrix_act.loc[m1, m2] += 1
                if not pd.isna(score1_tes) and not pd.isna(score2_tes):
                    if score1_tes > score2_tes: win_matrix_tes.loc[m1, m2] += 1
                    elif score2_tes > score1_tes: win_matrix_tes.loc[m2, m1] += 1
                    else: tie_matrix_tes.loc[m1, m2] += 1
    
    def calculate_win_details(win_matrix, tie_matrix):
        results = {m1: {} for m1 in models}
        rate_matrix = pd.DataFrame(index=models, columns=models, data=0.0)
        for m1 in models:
            for m2 in models:
                if m1 == m2: continue
                win_count = win_matrix.loc[m1, m2]
                loss_count = win_matrix.loc[m2, m1]
                
                # Get tie counts for the canonical pair
                canonical_pair = tuple(sorted((m1, m2)))
                num_ties = tie_matrix.loc[canonical_pair[0], canonical_pair[1]]
                
                total_comparisons = win_count + loss_count + num_ties
                total_trials = win_count + loss_count  # Number of non-tied comparisons

                p_val = np.nan
                if total_trials > 0:
                    # Perform exact sign test using binomtest
                    res = stats.binomtest(k=win_count, n=total_trials, p=0.5)
                    p_val = res.pvalue

                win_rate = win_count / total_comparisons if total_comparisons > 0 else 0
                ci_lower, ci_upper = proportion_confint(count=win_count, nobs=total_comparisons, method='wilson')
                results[m1][m2] = {'win_rate': win_rate, 'ci_lower': ci_lower, 'ci_upper': ci_upper, 'p_value': p_val}
                rate_matrix.loc[m1, m2] = win_rate
        return results, rate_matrix

    win_details_act, win_rate_act_df = calculate_win_details(win_matrix_act, tie_matrix_act)
    win_details_tes, win_rate_tes_df = calculate_win_details(win_matrix_tes, tie_matrix_tes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(win_rate_act_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5, ax=ax1, vmin=0, vmax=1)
    ax1.set_title('Head-to-Head Win Rate Matrix (ACT-FM)'); ax1.set_xlabel('Opponent Model'); ax1.set_ylabel('Model')
    sns.heatmap(win_rate_tes_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5, ax=ax2, vmin=0, vmax=1)
    ax2.set_title('Head-to-Head Win Rate Matrix (TES)'); ax2.set_xlabel('Opponent Model'); ax2.set_ylabel('Model')
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'subgroup_win_rate_matrices.png')); plt.close()

    return win_details_act, win_details_tes

def run_feature_importance_analysis(df, target_metric='act_fm_total'):
    logging.info(f"--- Running Feature Importance Analysis for {target_metric} ---")
    features = [
        'archetype_name', 'psych_mindedness_level', 'interaction_style_name', 'age_lower', 
        'gender', 'occupation', 'primary_issue', 'life_event', 'coping_mechanism', 
        'relationship_status', 'support_system', 'model'
    ]
    df_analysis = df.dropna(subset=[target_metric, 'session_id']).copy()
    X = df_analysis[[f for f in features if f in df.columns]].copy()
    y = df_analysis[target_metric]
    
    for col in X.select_dtypes(include=np.number).columns: 
        X[col].fillna(X[col].median(), inplace=True)
    X_encoded = pd.get_dummies(X, drop_first=True, dummy_na=True)
    
    groups = df_analysis['session_id']
    gkf = GroupKFold(n_splits=5)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    all_fold_importances = []
    oof_y_true = []
    oof_y_pred = []
    
    for train_idx, test_idx in gkf.split(X_encoded, y, groups):
        X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        rf.fit(X_train, y_train)
        
        # Generate predictions and store for OOF performance calculation
        predictions = rf.predict(X_test)
        oof_y_true.extend(y_test)
        oof_y_pred.extend(predictions)
        
        perm_result = permutation_importance(
            rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        all_fold_importances.append(perm_result.importances_mean)
        
    # Calculate overall out-of-fold performance metrics
    oof_r2 = r2_score(oof_y_true, oof_y_pred)
    oof_mae = mean_absolute_error(oof_y_true, oof_y_pred)

    avg_importances = np.mean(all_fold_importances, axis=0)
    
    importances_df = pd.DataFrame({
        'feature': X_encoded.columns, 
        'importance': avg_importances
    }).sort_values('importance', ascending=False)
    
    return importances_df, oof_r2, oof_mae

def generate_subgroup_report(df):
    logging.info("--- Generating Subgroup Analysis Summary Report ---")
    report_lines = ["="*80, "EXPLORATORY SUBGROUP ANALYSIS REPORT", "="*80 + "\n"]

    optimal_models = {}
    # Expanded list for optimal model analysis
    all_characteristics = [
        'archetype_name', 'psych_mindedness_level', 'interaction_style_name', 
        'primary_issue', 'life_event', 'coping_mechanism', 'relationship_status', 'support_system'
    ]
    
    for char in all_characteristics:
        if char not in df.columns or df[char].nunique() < 2: continue
        char_results = {}
        for val in df[char].unique():
            subset = df[df[char] == val]
            if subset.empty: continue
            act_best = subset.groupby('model')['act_fm_total'].mean().idxmax()
            tes_best = subset.groupby('model')['tes_mean'].mean().idxmax()
            char_results[val] = {'best_act_fm': act_best, 'best_tes': tes_best}
        optimal_models[char] = char_results

    for char, results in optimal_models.items():
        report_lines.append(f"### Optimal Models by {char.replace('_', ' ').title()} ###")
        for val, scores in results.items():
            report_lines.append(f"  - For '{val}':")
            report_lines.append(f"    - Best on ACT-FM: {scores['best_act_fm']}")
            report_lines.append(f"    - Best on TES:    {scores['best_tes']}")
        report_lines.append("")
    
    report_lines.append("\n### COT Impact Analysis by Subgroup ###")
    report_lines.append("Difference in mean score (COT model - non-COT model)")
    model_families = ['Instruct', 'SFT', 'ORPO']
    for char in all_characteristics:
        if char not in df.columns: continue
        report_lines.append(f"\n--- By {char.replace('_', ' ').title()} ---")
        for val in df[char].dropna().unique():
            report_lines.append(f"  Subgroup: {val}")
            subset = df[df[char] == val]
            for family in model_families:
                cot_model = f"{family} (COT)"
                nocot_model = f"{family} (no COT)"
                if cot_model in subset['model'].values and nocot_model in subset['model'].values:
                    mean_cot_act = subset[subset['model'] == cot_model]['act_fm_total'].mean()
                    mean_nocot_act = subset[subset['model'] == nocot_model]['act_fm_total'].mean()
                    mean_cot_tes = subset[subset['model'] == cot_model]['tes_mean'].mean()
                    mean_nocot_tes = subset[subset['model'] == nocot_model]['tes_mean'].mean()
                    
                    if pd.notna(mean_cot_act) and pd.notna(mean_nocot_act):
                        diff_act = mean_cot_act - mean_nocot_act
                        report_lines.append(f"    - {family} | ACT-FM Diff: {diff_act:+.2f}")
                    if pd.notna(mean_cot_tes) and pd.notna(mean_nocot_tes):
                        diff_tes = mean_cot_tes - mean_nocot_tes
                        report_lines.append(f"    - {family} | TES Diff:    {diff_tes:+.2f}")
    report_lines.append("")

    report_lines.append("### Statistically Significant Differences (FDR Corrected p < 0.05) ###")
    for char in all_characteristics:
        if char not in df.columns or df[char].nunique() < 2: continue
        comparisons = analyze_model_differences_by_characteristic(df, char, 'act_fm_total')
        sig_diffs = []
        for val, res in comparisons.items():
            for comp in res.get('comparisons', []):
                if comp.get('significant_adj', False):
                    sig_diffs.append(f"  - {val}: {comp['model1']} vs {comp['model2']} (p_adj={comp['p_adj']:.4f})")
        if sig_diffs:
            report_lines.append(f"\nFor {char.replace('_', ' ').title()}:")
            report_lines.extend(sig_diffs)

    with open(SUBGROUP_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    logging.info(f"Subgroup analysis report saved to '{SUBGROUP_REPORT_FILE}'")


# --- Part 5: Mixed-Effects Primary And Exploratory Analyses ---

def fit_mixedlm_random_intercept(df: pd.DataFrame, dependent_variable: str):
    data = df.copy()
    data = data.dropna(subset=[dependent_variable, 'model', 'session_id'])
    data['session_id'] = data['session_id'].astype(str)
    data['model'] = pd.Categorical(data['model'])
    formula = f"{dependent_variable} ~ C(model)"
    model = smf.mixedlm(formula=formula, data=data, groups=data['session_id'])
    result = model.fit(method='lbfgs', reml=True)
    return result, data

def calculate_icc_with_bootstrap(df: pd.DataFrame, dependent_variable: str, n_bootstraps: int = 500):
    logging.info(f"  Calculating ICC with bootstrap for {dependent_variable} ({n_bootstraps} iterations)...")
    data = df.dropna(subset=[dependent_variable, 'model', 'session_id']).copy()
    data['session_id'] = data['session_id'].astype(str)
    data['model'] = pd.Categorical(data['model'])

    session_ids = data['session_id'].unique()
    icc_values = []

    for i in range(n_bootstraps):
        bootstrap_session_ids = np.random.choice(session_ids, size=len(session_ids), replace=True)
        
        # Create a mapping from original to new IDs to avoid duplicate group labels
        id_map = {orig_id: f"{orig_id}_{j}" for j, orig_id in enumerate(bootstrap_session_ids)}
        
        # Build the bootstrap sample
        bootstrap_df = pd.concat([
            data[data['session_id'] == orig_id].assign(bootstrap_session_id=new_id)
            for orig_id, new_id in id_map.items()
        ])
        
        if bootstrap_df['bootstrap_session_id'].nunique() < 2:
            continue

        try:
            formula = f"{dependent_variable} ~ C(model)"
            model = smf.mixedlm(formula=formula, data=bootstrap_df, groups=bootstrap_df['bootstrap_session_id'])
            result = model.fit(method='lbfgs', reml=True, disp=False)

            if result.converged:
                group_var = float(np.asarray(result.cov_re)[0, 0])
                residual_var = float(result.scale)
                if (group_var + residual_var) > 0:
                    icc = group_var / (group_var + residual_var)
                    icc_values.append(icc)
        except Exception:
            continue
    
    if not icc_values:
        logging.warning(f"Bootstrap for ICC failed for {dependent_variable}, not enough successful fits.")
        return np.nan, np.nan

    lower_bound = np.percentile(icc_values, 2.5)
    upper_bound = np.percentile(icc_values, 97.5)
    
    return lower_bound, upper_bound

def summarize_fixed_and_random_effects(result: sm.regression.mixed_linear_model.MixedLMResults, dependent_variable: str):
    fe = result.fe_params
    bse = result.bse_fe
    z_vals = fe / bse
    p_vals = 2 * stats.norm.sf(np.abs(z_vals))
    ci_low = fe - 1.96 * bse
    ci_high = fe + 1.96 * bse
    fixed_df = pd.DataFrame({
        'term': fe.index,
        'estimate': fe.values,
        'std_error': bse.values,
        'z_value': z_vals.values,
        'p_value': p_vals,
        'ci_lower_95': ci_low.values,
        'ci_upper_95': ci_high.values
    })
    var_components = []
    if result.cov_re is not None:
        var_components.append({'component': 'session_id_intercept', 'variance': float(np.asarray(result.cov_re)[0, 0])})
    var_components.append({'component': 'residual', 'variance': float(result.scale)})
    random_df = pd.DataFrame(var_components)
    summary_text_path = os.path.join(TABLES_DIR, f"mixedlm_summary_{dependent_variable}.txt")
    with open(summary_text_path, 'w', encoding='utf-8') as f:
        f.write(str(result.summary()))
    logging.info(f"Saved MixedLM full summary for {dependent_variable} to '{summary_text_path}'")
    logging.info(f"\nMixedLM Fixed Effects ({dependent_variable}):\n{fixed_df.round(4)}")
    logging.info(f"\nMixedLM Variance Components ({dependent_variable}):\n{random_df.round(4)}")
    fixed_path = os.path.join(TABLES_DIR, f"mixedlm_fixed_effects_{dependent_variable}.csv")
    random_path = os.path.join(TABLES_DIR, f"mixedlm_variance_components_{dependent_variable}.csv")
    fixed_df.to_csv(fixed_path, index=False)
    random_df.to_csv(random_path, index=False)
    return fixed_df, random_df

def pairwise_comparisons_mixedlm(result: sm.regression.mixed_linear_model.MixedLMResults, data: pd.DataFrame, dependent_variable: str):
    levels = list(data['model'].cat.categories)
    fe_names = list(result.fe_params.index)

    full_cov_matrix = result.cov_params()
    cov = full_cov_matrix.loc[fe_names, fe_names]

    def design_for_level(level):
        vec = np.zeros(len(fe_names))
        if 'Intercept' in fe_names:
            vec[fe_names.index('Intercept')] = 1.0
        term = f"C(model)[T.{level}]"
        if term in fe_names:
            vec[fe_names.index(term)] = 1.0
        return vec
    rows = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            a, b = levels[i], levels[j]
            Xa = design_for_level(a)
            Xb = design_for_level(b)
            contrast = Xa - Xb
            diff = float(contrast @ result.fe_params.values)
            se = float(np.sqrt(contrast @ cov.values @ contrast))
            if se == 0:
                z = np.nan
                p = np.nan
                ci_lo, ci_hi = np.nan, np.nan
            else:
                z = diff / se
                p = 2 * stats.norm.sf(abs(z))
                ci_lo = diff - 1.96 * se
                ci_hi = diff + 1.96 * se
            rows.append({
                'dependent_variable': dependent_variable,
                'group1': a,
                'group2': b,
                'mean_difference': diff,
                'std_error': se,
                'z_value': z,
                'p_value_raw': p,
                'ci_lower_95': ci_lo,
                'ci_upper_95': ci_hi
            })
    results_df = pd.DataFrame(rows)
    if not results_df.empty and 'p_value_raw' in results_df.columns and not results_df['p_value_raw'].isnull().all():
        pvals_non_nan = results_df['p_value_raw'].dropna()
        if not pvals_non_nan.empty:
            reject, p_adj, _, _ = multipletests(pvals_non_nan, alpha=ALPHA, method='holm-sidak')
            p_adj_series = pd.Series(p_adj, index=pvals_non_nan.index)
            reject_series = pd.Series(reject, index=pvals_non_nan.index)
            results_df['p_value_adjusted'] = p_adj_series
            results_df['reject_null'] = reject_series
        results_df['adjustment_method'] = 'Holm-Sidak'
    else:
        results_df['p_value_adjusted'] = np.nan
        results_df['reject_null'] = np.nan
        results_df['adjustment_method'] = 'Holm-Sidak'

    out_path = os.path.join(TABLES_DIR, f"mixedlm_posthoc_{dependent_variable}.csv")
    results_df.to_csv(out_path, index=False)
    logging.info(f"Saved MixedLM post-hoc pairwise comparisons for {dependent_variable} to '{out_path}'")
    return results_df

def run_mixed_effects_primary(df: pd.DataFrame):
    logging.info("--- Running Primary Analysis (Linear Mixed-Effects Models) ---")
    all_posthoc = []
    all_results = {}
    for dependent_variable in DEPENDENT_VARIABLES:
        result, data_used = fit_mixedlm_random_intercept(df, dependent_variable)
        all_results[dependent_variable] = result

        # Calculate R-squared values for the mixed model
        var_f = np.var(np.dot(result.model.exog, result.fe_params))
        var_a = float(np.asarray(result.cov_re)[0, 0])
        var_e = float(result.scale)
        total_var = var_f + var_a + var_e

        if total_var > 0:
            marginal_r2 = var_f / total_var
            conditional_r2 = (var_f + var_a) / total_var
        else:
            marginal_r2, conditional_r2 = np.nan, np.nan
        
        logging.info(f"\nR-squared for {dependent_variable}:")
        logging.info(f"  Marginal R² (fixed effects): {marginal_r2:.4f}")
        logging.info(f"  Conditional R² (fixed + random effects): {conditional_r2:.4f}")

        r_squared_results = {
            "marginal_r2": marginal_r2,
            "conditional_r2": conditional_r2
        }
        
        # 1. Omnibus Wald Chi-squared Test
        model_param_names = [name for name in result.fe_params.index if name.startswith("C(model)")]
        wald_test = result.wald_test(model_param_names)
        chi2_statistic = wald_test.statistic[0][0]
        df_num = len(model_param_names) # The DF is the number of constraints (parameters tested)
        p_value = wald_test.pvalue
        logging.info(f"\nOmnibus Wald Test for 'model' on {dependent_variable}:")
        logging.info(f"  Chi-squared = {chi2_statistic:.4f}, df = {df_num}, p = {p_value:.4f}")

        omnibus_test_results = {
            "chi2_statistic": chi2_statistic,
            "df": df_num,
            "p_value": p_value
        }

        # 4. Estimated Marginal Means (EMMs)
        emm_results = []
        model_levels = data_used['model'].cat.categories
        fe_params = result.fe_params
        vcov = result.cov_params()
        
        # The first level is the reference (intercept)
        ref_level = model_levels[0]
        
        # Calculate EMM for the reference level
        mean_ref = fe_params['Intercept']
        se_ref = np.sqrt(vcov.loc['Intercept', 'Intercept'])
        ci_lower_ref = mean_ref - 1.96 * se_ref
        ci_upper_ref = mean_ref + 1.96 * se_ref
        emm_results.append({
            "model_name": ref_level,
            "emm": mean_ref,
            "ci_lower": ci_lower_ref,
            "ci_upper": ci_upper_ref
        })

        # Calculate EMM for all other levels
        for level in model_levels[1:]:
            param_name = f"C(model)[T.{level}]"
            
            # Predicted mean is Intercept + level effect
            mean_level = fe_params['Intercept'] + fe_params[param_name]
            
            # Variance of the sum of two correlated variables (Intercept and level effect)
            var_mean = (vcov.loc['Intercept', 'Intercept'] + 
                        vcov.loc[param_name, param_name] + 
                        2 * vcov.loc['Intercept', param_name])
            se_level = np.sqrt(var_mean)
            
            ci_lower_level = mean_level - 1.96 * se_level
            ci_upper_level = mean_level + 1.96 * se_level
            
            emm_results.append({
                "model_name": level,
                "emm": mean_level,
                "ci_lower": ci_lower_level,
                "ci_upper": ci_upper_level
            })

        logging.info(f"\nEstimated Marginal Means for {dependent_variable}:")
        for emm in emm_results:
            logging.info(f"  - {emm['model_name']}: {emm['emm']:.2f} (95% CI: {emm['ci_lower']:.2f}, {emm['ci_upper']:.2f})")

        # 5. ICC Calculation with Bootstrap CI
        group_var = float(np.asarray(result.cov_re)[0, 0])
        residual_var = float(result.scale)
        icc_point_estimate = group_var / (group_var + residual_var) if (group_var + residual_var) > 0 else 0
        icc_ci_lower, icc_ci_upper = calculate_icc_with_bootstrap(df, dependent_variable)
        
        icc_results = {
            "icc": icc_point_estimate,
            "icc_ci_lower": icc_ci_lower,
            "icc_ci_upper": icc_ci_upper
        }
        logging.info(f"\nIntraclass Correlation (ICC) for {dependent_variable}:")
        logging.info(f"  ICC = {icc_point_estimate:.3f} (95% Bootstrap CI: {icc_ci_lower:.3f}, {icc_ci_upper:.3f})")

        logging.info(f"\nFull MixedLM Summary for {dependent_variable}:\n{result.summary()}")
        fixed_df, random_df = summarize_fixed_and_random_effects(result, dependent_variable)
        posthoc = pairwise_comparisons_mixedlm(result, data_used, dependent_variable)

        # 6. Diagnostic Plots
        residuals = result.resid
        fitted = result.fittedvalues
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=fitted, y=residuals)
        plt.axhline(0, linestyle='--', color='red')
        plt.title(f'Residuals vs. Fitted Plot for {dependent_variable}')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.savefig(os.path.join(PLOTS_DIR, f'diagnostic_residuals_vs_fitted_{dependent_variable}.png'))
        plt.close()
        
        fig = sm.qqplot(residuals, line='s')
        plt.title(f'Q-Q Plot of Residuals for {dependent_variable}')
        plt.savefig(os.path.join(PLOTS_DIR, f'diagnostic_qqplot_{dependent_variable}.png'))
        plt.close()
        logging.info(f"Saved diagnostic plots for {dependent_variable}.")

        # Save all results to global JSON object
        analysis_results_json["primary_analysis"]["linear_mixed_effects_models"][dependent_variable] = {
            "description": f"Linear mixed-effects model with model type as a fixed effect and patient session ID as a random intercept, predicting {dependent_variable}.",
            "r_squared": r_squared_results,
            "omnibus_test": omnibus_test_results,
            "estimated_marginal_means": emm_results,
            "intraclass_correlation": icc_results,
            "fixed_effects": fixed_df.to_dict('records'),
            "random_effects_variance": random_df.to_dict('records'),
            "post_hoc_pairwise_comparisons": posthoc.to_dict('records')
        }

        if not posthoc.empty:
            table_ready = posthoc[['dependent_variable', 'group1', 'group2', 'mean_difference', 'p_value_adjusted', 'ci_lower_95', 'ci_upper_95']].copy()
            table_ready.rename(columns={
                'dependent_variable': 'Test',
                'group1': 'Level A',
                'group2': 'Level B',
                'mean_difference': 'Statistic',
                'p_value_adjusted': 'p-value'
            }, inplace=True)
            table_ready['Test'] = table_ready['Test'].apply(lambda x: f"MixedLM Pairwise ({x})")
            table_ready['F/H'] = ''
            table_ready['Significant'] = table_ready['p-value'].apply(lambda p: 'Yes' if p < ALPHA else 'No')
            all_posthoc.append(table_ready[['Test', 'Level A', 'Level B', 'Statistic', 'F/H', 'p-value', 'Significant', 'ci_lower_95', 'ci_upper_95']])

    if all_posthoc:
        return pd.concat(all_posthoc, ignore_index=True), all_results
    return None, all_results

def run_mixed_effects_interaction(df: pd.DataFrame):
    logging.info("--- Running Exploratory Interaction Analysis (Model × Archetype) ---")
    interaction_summaries = {}
    for dependent_variable in DEPENDENT_VARIABLES:
        data = df.dropna(subset=[dependent_variable, 'model', 'session_id', 'archetype_name']).copy()
        if data.empty:
            logging.warning(f"No data available for interaction model for {dependent_variable}. Skipping.")
            continue
        data['session_id'] = data['session_id'].astype(str)
        data['model'] = pd.Categorical(data['model'])
        data['archetype_name'] = pd.Categorical(data['archetype_name'])
        formula = f"{dependent_variable} ~ C(model) * C(archetype_name)"
        try:
            model = smf.mixedlm(formula=formula, data=data, groups=data['session_id'])
            result = model.fit(method='lbfgs', reml=False)
            summary_text = str(result.summary())
            logging.info(f"\nFull MixedLM Interaction Summary for {dependent_variable}:\n{summary_text}")
            interaction_summaries[dependent_variable] = summary_text
            out_path = os.path.join(TABLES_DIR, f"mixedlm_interaction_summary_{dependent_variable}.txt")
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            logging.info(f"Saved MixedLM interaction summary for {dependent_variable} to '{out_path}'")
        except Exception as e:
            logging.error(f"Failed to fit interaction model for {dependent_variable}: {e}")
            interaction_summaries[dependent_variable] = f"ERROR: Failed to fit model - {e}"
    
    analysis_results_json["exploratory_subgroup_analysis"]["interaction_analysis"] = {
        "description": "Exploratory linear mixed-effects models including an interaction term between therapist model type and patient archetype to examine whether model effectiveness varies systematically across patient types.",
        "summaries": interaction_summaries
    }

# --- Main Orchestration ---

def main():
    for path in [OUTPUT_DIR, PLOTS_DIR, TABLES_DIR]:
        os.makedirs(path, exist_ok=True)
    logging.info(f"Analysis outputs will be saved in '{OUTPUT_DIR}'")

    if os.path.exists(PRE_EXISTING_CSV_PATH):
        logging.info(f"Found pre-existing CSV at '{PRE_EXISTING_CSV_PATH}'. Copying to output directory.")
        shutil.copy(PRE_EXISTING_CSV_PATH, PARSED_EVALS_CSV)
    elif not os.path.exists(PARSED_EVALS_CSV):
        logging.info(f"'{PARSED_EVALS_CSV}' not found. Generating from raw evaluation files...")
        if not generate_csv_from_evaluations():
            logging.error("Failed to generate CSV. Exiting.")
            return
    else:
        logging.info(f"Found existing parsed evaluation file: '{PARSED_EVALS_CSV}'")

    df_unprocessed = load_and_merge_data()
    if df_unprocessed is None: return

    df_processed = process_and_engineer_features(df_unprocessed.copy())

    generate_exclusion_report(df_processed, RAW_TRANSCRIPT_DIRS, EXCLUSION_REPORT_FILE)

    logging.info("--- Cleaning Data for Analysis ---")
    logging.info(f"Initial rows loaded and merged: {len(df_processed)}")
    valid_statuses = ['OK', 'PARTIAL']
    df_clean = df_processed[df_processed['parsing_status'].isin(valid_statuses)].copy()
    rows_after_status_filter = len(df_clean)
    logging.info(f"Rows after filtering for valid parse statuses: {rows_after_status_filter}")
    
    original_rows = len(df_clean)
    tes_items = [f'tes_item_{i}' for i in range(1, 10)]
    act_fm_items = [f'act_fm_item_{i}' for i in range(1, 26)]
    df_clean.dropna(subset=tes_items + act_fm_items, inplace=True)
    rows_after_na_drop = len(df_clean)
    logging.info(f"Rows after dropping sessions with any missing item scores: {rows_after_na_drop} ({original_rows - rows_after_na_drop} removed)")

    # Store data summary in global JSON object
    analysis_results_json["data_summary"] = {
        "initial_records_loaded": len(df_processed),
        "records_after_filtering_parse_status": rows_after_status_filter,
        "final_records_for_analysis": rows_after_na_drop,
        "records_excluded_total": len(df_processed) - rows_after_na_drop,
        "models_compared": sorted(df_clean['model'].unique())
    }

    if len(df_clean['model'].unique()) < 2 or len(df_clean) < 10:
        logging.error("Insufficient valid data remains after cleaning. Aborting analysis.")
        return

    logging.info("\n" + "="*25 + " STARTING PRIMARY ANALYSIS (MIXED EFFECTS) " + "="*25)
    desc_stats_df = df_clean.groupby('model')[DEPENDENT_VARIABLES].agg(['mean', 'std', 'count']).round(3)
    logging.info(f"\nDescriptive Statistics:\n{desc_stats_df}")
    
    # Store descriptive statistics in global JSON object
    desc_stats_df_reset = desc_stats_df.copy()
    desc_stats_df_reset.columns = ['_'.join(col).strip() for col in desc_stats_df_reset.columns.values]
    analysis_results_json["descriptive_statistics"] = {
        "description": "Descriptive statistics (Mean, Standard Deviation, Count) for the primary outcome measures, grouped by model type.",
        "statistics": desc_stats_df_reset.reset_index().to_dict('records')
    }
    
    create_primary_visualizations(df_clean)
    mixedlm_table2, mixedlm_results = run_mixed_effects_primary(df_clean)
    
    create_publication_tables(df_clean, mixedlm_table2)

    if mixedlm_table2 is not None:
        for dv in DEPENDENT_VARIABLES:
            create_forest_plot_from_mixedlm(mixedlm_table2, dv, PLOTS_DIR)

    logging.info("\n" + "="*20 + " STARTING EXPLORATORY SUBGROUP ANALYSIS " + "="*20)
    create_subgroup_visualizations(df_clean)
    win_details_act, win_details_tes = create_win_rate_matrix(df_clean)
    
    # Store win rates in global JSON object
    analysis_results_json["exploratory_subgroup_analysis"]["head_to_head_win_rates"] = {
        "description": "Head-to-head win rates for each pair of models based on which achieved a higher score for a given patient profile. Values represent the proportion of sessions won, with 95% Wilson confidence intervals.",
        "act_fm_total": win_details_act,
        "tes_mean": win_details_tes
    }
    
    fi_act, oof_r2_act, oof_mae_act = run_feature_importance_analysis(df_clean, 'act_fm_total')
    fi_tes, oof_r2_tes, oof_mae_tes = run_feature_importance_analysis(df_clean, 'tes_mean')

    logging.info(f"\nRandom Forest OOF R² (predicting ACT-FM): {oof_r2_act:.3f}")
    logging.info(f"Random Forest OOF MAE (predicting ACT-FM): {oof_mae_act:.3f}")
    logging.info(f"Top 5 Features for ACT-FM Score:\n{fi_act.head().to_string(index=False)}")
    create_feature_importance_plot(fi_act, 'act_fm_total', PLOTS_DIR)

    logging.info(f"\nRandom Forest OOF R² (predicting TES): {oof_r2_tes:.3f}")
    logging.info(f"Random Forest OOF MAE (predicting TES): {oof_mae_tes:.3f}")
    logging.info(f"Top 5 Features for TES Score:\n{fi_tes.head().to_string(index=False)}")
    create_feature_importance_plot(fi_tes, 'tes_mean', PLOTS_DIR)

    # Store feature importance in global JSON object
    analysis_results_json["exploratory_subgroup_analysis"]["feature_importance"] = {
        "description": "Feature importances from a Random Forest Regressor model trained to predict scores based on patient characteristics and therapist model type. Calculated using permutation importance on a test set from a 5-fold GroupKFold cross-validation.",
        "predicting_act_fm_score": {
            "model_performance": {"r2": oof_r2_act, "mae": oof_mae_act},
            "importances": fi_act.to_dict('records')
        },
        "predicting_tes_score": {
            "model_performance": {"r2": oof_r2_tes, "mae": oof_mae_tes},
            "importances": fi_tes.to_dict('records')
        }
    }

    generate_subgroup_report(df_clean)
    
    # Capture optimal models and COT impact for JSON
    optimal_models_json = {}
    all_characteristics_report = [
        'archetype_name', 'psych_mindedness_level', 'interaction_style_name', 'primary_issue',
        'life_event', 'coping_mechanism', 'relationship_status', 'support_system'
    ]
    for char in all_characteristics_report:
        if char not in df_clean.columns or df_clean[char].nunique() < 2: continue
        optimal_models_json[char] = {}
        for val in df_clean[char].unique():
            subset = df_clean[df_clean[char] == val]
            if subset.empty: continue
            act_best = subset.groupby('model')['act_fm_total'].mean().idxmax()
            tes_best = subset.groupby('model')['tes_mean'].mean().idxmax()
            optimal_models_json[char][val] = {'best_model_act_fm': act_best, 'best_model_tes': tes_best}
    analysis_results_json["exploratory_subgroup_analysis"]["optimal_model_by_subgroup"] = optimal_models_json

    cot_impact_json = {}
    model_families = ['Instruct', 'SFT', 'ORPO']
    for char in all_characteristics_report:
        if char not in df_clean.columns: continue
        cot_impact_json[char] = {}
        for val in df_clean[char].dropna().unique():
            cot_impact_json[char][val] = {}
            subset = df_clean[df_clean[char] == val]
            for family in model_families:
                cot_model = f"{family} (COT)"; nocot_model = f"{family} (no COT)"
                if cot_model in subset['model'].values and nocot_model in subset['model'].values:
                    diff_act = subset[subset['model'] == cot_model]['act_fm_total'].mean() - subset[subset['model'] == nocot_model]['act_fm_total'].mean()
                    diff_tes = subset[subset['model'] == cot_model]['tes_mean'].mean() - subset[subset['model'] == nocot_model]['tes_mean'].mean()
                    cot_impact_json[char][val][family] = {'act_fm_difference': diff_act, 'tes_difference': diff_tes}
    analysis_results_json["exploratory_subgroup_analysis"]["cot_impact_analysis"] = cot_impact_json


    logging.info("\n" + "="*20 + " STARTING EXPLORATORY INTERACTION ANALYSIS (MIXED EFFECTS) " + "="*20)
    run_mixed_effects_interaction(df_clean)

    # --- Save LLM-Friendly JSON Output ---
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if pd.isna(obj):
                return None
            return super(NpEncoder, self).default(obj)

    logging.info(f"\nSaving structured analysis results to '{JSON_OUTPUT_FILE}'...")
    with open(JSON_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(analysis_results_json, f, indent=4, cls=NpEncoder)
    logging.info("JSON summary saved successfully.")

    logging.info("\n" + "="*60)
    logging.info("UNIFIED ANALYSIS COMPLETE")
    logging.info(f"All outputs have been saved to: {OUTPUT_DIR}")
    logging.info("="*60)

if __name__ == '__main__':
    main()