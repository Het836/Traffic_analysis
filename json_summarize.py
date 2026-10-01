import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path("./traffic_data.json")
OUTPUT_PATH = Path("./traffic_data_summary.json")

# True  -> Local Qwen GGUF
# False -> API model
USE_LOCAL_MODEL = False

# Used when USE_LOCAL_MODEL = False
# "groq", "gemini", or "mistral"
MODEL_PROVIDER = "groq"

# Used when USE_LOCAL_MODEL = True
MODEL_PATH = (
    "/home/nvidia/jetson_traffic_llm/models/"
    "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)


# ============================================================
# LOAD MODEL
# ============================================================

def get_llm():

    if USE_LOCAL_MODEL:

        from langchain_community.llms import LlamaCpp

        print("Using local Qwen model")

        return LlamaCpp(
            model_path=MODEL_PATH,
            temperature=0,
            max_tokens=512,
            verbose=True
        )

    if MODEL_PROVIDER == "groq":

        print("Using Groq")

        return ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0
        )

    elif MODEL_PROVIDER == "gemini":

        print("Using Gemini")

        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )

    elif MODEL_PROVIDER == "mistral":

        print("Using Mistral")

        return ChatMistralAI(
            model="mistral-small-latest",
            temperature=0
        )

    else:
        raise ValueError(
            "MODEL_PROVIDER must be "
            "'groq', 'gemini', or 'mistral'"
        )


llm = get_llm()


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a traffic data analysis system.

Analyze the provided traffic records for a 10-second interval.

Return ONLY a valid JSON object.

Do NOT use markdown.
Do NOT use ```json.
Do NOT add explanations.

Use exactly this structure:

{{
    "summary": "short natural language traffic summary",
    "vehicle_summary": {{
        "total": 0,
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }},
    "average_speed_kmh": 0.0,
    "incident": false,
    "traffic_density": "Low"
}}

Rules:

1. vehicle_summary must contain the SUM of vehicle counts
   across all records.

2. average_speed_kmh must be the arithmetic average of
   speed_kmh values greater than 0.

3. incident is true if ANY record has
   incident.detected == true.

4. traffic_density must be exactly:
   "Low", "Medium", or "High".

5. Consider all records in the interval.

6. The summary should mention:
   - vehicle activity
   - approximate speed
   - traffic density
   - incident status

7. Do not invent information.

8. Return ONLY the JSON object.
"""
    ),
    (
        "human",
        """
Analyze these traffic records:

{traffic_data}
"""
    )
])


chain = prompt | llm


# ============================================================
# LOAD JSON
# ============================================================

def load_data():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# TIME CONVERSION
# ============================================================

def time_to_seconds(time_string):

    minutes, seconds = time_string.split(":")

    return (
        int(minutes) * 60
        + float(seconds)
    )


# ============================================================
# GROUP DATA INTO 10-SECOND WINDOWS
# ============================================================

def group_every_10_seconds(data):

    groups = {}

    for record in data:

        seconds = time_to_seconds(
            record["time"]
        )

        window = int(seconds // 10)

        if window not in groups:
            groups[window] = []

        groups[window].append(record)

    return groups


# ============================================================
# CLEAN MODEL RESPONSE
# ============================================================

def clean_json_response(response):

    if hasattr(response, "content"):
        response = response.content

    response = str(response).strip()

    if response.startswith("```json"):
        response = response[7:]

    elif response.startswith("```"):
        response = response[3:]

    if response.endswith("```"):
        response = response[:-3]

    response = response.strip()

    first_brace = response.find("{")
    last_brace = response.rfind("}")

    if first_brace != -1 and last_brace != -1:

        response = response[
            first_brace:last_brace + 1
        ]

    return response


# ============================================================
# SUMMARIZE ONE 10-SECOND WINDOW
# ============================================================

def summarize_window(records, window_number):

    start_seconds = window_number * 10
    end_seconds = start_seconds + 10

    start_time = (
        f"{int(start_seconds // 60):02d}:"
        f"{start_seconds % 60:04.1f}"
    )

    end_time = (
        f"{int(end_seconds // 60):02d}:"
        f"{end_seconds % 60:04.1f}"
    )

    time_range = f"{start_time} - {end_time}"

    traffic_data = json.dumps(
        records,
        indent=2
    )

    response = chain.invoke({
        "traffic_data": traffic_data
    })

    response_text = clean_json_response(
        response
    )

    try:

        result = json.loads(
            response_text
        )

    except json.JSONDecodeError as e:

        print("\nInvalid JSON returned by model:")
        print(response_text)

        raise ValueError(
            f"Could not parse response for "
            f"{time_range}"
        ) from e

    result["time_range"] = time_range

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("Traffic Analysis")
    print("=" * 60)

    print("\nLoading traffic data...")

    data = load_data()

    print(
        f"Loaded {len(data)} records"
    )

    groups = group_every_10_seconds(
        data
    )

    print(
        f"Created {len(groups)} time windows"
    )

    summaries = []

    for window_number, records in groups.items():

        print()

        print(
            f"Processing "
            f"{window_number * 10}s - "
            f"{window_number * 10 + 10}s"
        )

        print(
            f"Records: {len(records)}"
        )

        summary = summarize_window(
            records,
            window_number
        )

        summaries.append(summary)

        print("Summary generated.")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summaries,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("Done!")
    print(
        f"Output saved to: {OUTPUT_PATH}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()