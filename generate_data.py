import os
import re
import json
import asyncio
import logging
import hashlib
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
import ollama
import tiktoken
import fitz  # PyMuPDF
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Confirm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 0. Load Environment Variables & Initialize
# ==========================================
load_dotenv()

# Global Async Lock for safe file I/O operations (Fixes Race Conditions)
file_lock = asyncio.Lock()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
INPUT_DIR = Path(os.getenv("INPUT_DIR", "./data_input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./data_output"))
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
TARGET_YIELD = int(os.getenv("TARGET_YIELD", "500"))
AUTO_BYPASS = os.getenv("AUTO_BYPASS_WARNING", "False").lower() == "true"
GENERATION_STYLE = os.getenv("GENERATION_STYLE", "cot").lower()
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "English")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "auto").strip()
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "2"))
MIN_QUALITY_SCORE = int(os.getenv("MIN_QUALITY_SCORE", "4"))
CHECKPOINT_FILE = Path(os.getenv("CHECKPOINT_FILE", "./logs/state.json"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# Hardware & Token Optimization based on Generation Style
if GENERATION_STYLE == "cot":
    NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX_COT", "8192"))
    NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT_COT", "2500"))
    TOKENS_PER_PAIR = 400
else:
    NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX_QA", "4096"))
    NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT_QA", "1000"))
    TOKENS_PER_PAIR = 150

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure Logging
logging.basicConfig(
    filename=LOG_DIR / "pipeline_run.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

console = Console()

# Secure Client Initialization
client_kwargs = {"host": OLLAMA_HOST}
OLLAMA_SECRET = os.getenv("OLLAMA_SECRET", "").strip()
if OLLAMA_SECRET:
    client_kwargs["headers"] = {"Authorization": f"Bearer {OLLAMA_SECRET}"}

client = ollama.AsyncClient(**client_kwargs)

# ==========================================
# 1. Define Pydantic Validation Models
# ==========================================
class DirectQAPair(BaseModel):
    instruction: str
    output: str

class CoTPair(BaseModel):
    instruction: str
    reasoning: str
    output: str

class JudgeScore(BaseModel):
    score: int

def get_schema():
    return CoTPair if GENERATION_STYLE == "cot" else DirectQAPair

# ==========================================
# 2. Core Utility Functions
# ==========================================
def extract_and_clean_text() -> str:
    """Phase 1: Universal PDF processing with multi-document boundaries and fault tolerance."""
    full_text = ""
    for pdf_path in INPUT_DIR.glob("*.pdf"):
        try:
            with fitz.open(pdf_path) as doc:
                full_text += f"\n\n--- Document: {pdf_path.name} ---\n\n"
                for page in doc:
                    text = page.get_text()
                    text = re.sub(r'(?i)^\s*page\s*\d+\s*$', '', text, flags=re.MULTILINE)
                    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
                    full_text += text + "\n"
        except Exception as e:
            console.print(f"[red]Warning: Failed to read {pdf_path.name}: {e}[/red]")
            logging.warning(f"Failed to read {pdf_path.name}: {e}")
            
    return full_text

def get_chunk_hash(text: str) -> str:
    """Generate a unique Hash ID for a text chunk."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def load_checkpoints() -> dict:
    """Load checkpoint state for resuming interrupted processes (Zero-byte resilient)."""
    if CHECKPOINT_FILE.exists() and CHECKPOINT_FILE.stat().st_size > 0:
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Checkpoint file {CHECKPOINT_FILE} is corrupted. Resetting state.")
            return {}
    return {}

def save_checkpoint(state: dict):
    """Save the current checkpoint state."""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def extract_json(response_text: str) -> str:
    """Omni-Directional JSON Extractor for handling arrays and objects even without Markdown tags."""
    match = re.search(r'`{3}(?:json)?\s*([\s\S]*?)`{3}', response_text)
    if match:
        return match.group(1).strip()
    
    first_bracket, last_bracket = response_text.find('['), response_text.rfind(']')
    first_brace, last_brace = response_text.find('{'), response_text.rfind('}')
    
    starts = [idx for idx in (first_bracket, first_brace) if idx != -1]
    ends = [idx for idx in (last_bracket, last_brace) if idx != -1]
    
    if starts and ends:
        start_idx = min(starts)
        end_idx = max(ends)
        if end_idx > start_idx:
            return response_text[start_idx:end_idx+1]
            
    return response_text.strip()

# ==========================================
# 3. LLM Interaction & Processing Logic
# ==========================================
async def detect_domain_persona(text: str) -> str:
    """NEW Phase: Automatically detect the document domain with strict structural boundaries."""
    console.print("[cyan]Auto-detecting document domain and generating persona...[/cyan]")
    
    prompt = f"""
    You are an expert taxonomy AI. Analyze the <input_text> and determine its professional domain.
    Based on that, write EXACTLY ONE sentence that acts as a system prompt for an AI expert in this domain.
    Start the sentence with "You are an expert...". Do not output anything else.
    
    <example>
    <input_text>The patient presents with acute myocardial infarction and requires immediate ECG monitoring.</input_text>
    Output: You are an expert medical AI assistant specializing in cardiology and patient care.
    </example>
    
    <input_text>
    {text[:5000]}
    </input_text>
    
    Output:
    """
    try:
        response = await client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"num_ctx": NUM_CTX, "temperature": 0.1, "num_predict": 50}
        )
        persona = response['response'].strip().strip('"').strip("'")
        
        # Comprehensive text cleanup for small models that output extra formatting
        persona = re.sub(r'(?i)^(output|persona|system prompt):\s*', '', persona).strip()
        
        if not persona.startswith("You are"):
            # Try to extract from the first occurrence of "You are" if buried in chat clutter
            match = re.search(r'(You are .*?\.)', persona)
            if match:
                persona = match.group(1)
            else:
                persona = f"You are an expert assistant. {persona}"
            
        return persona
    except Exception as e:
        logging.error(f"Auto-persona detection failed: {e}")
        return "You are a helpful and highly knowledgeable expert assistant."

async def generate_global_context(text: str) -> str:
    """Phase 2: Generate the Global Context Map for cross-document awareness."""
    console.print("[cyan]Generating cross-document Global Context Map...[/cyan]")
    prompt = f"""
    As an expert AI Architect, analyze the following document excerpt. 
    Identify the core dependencies, potential trade-offs, and operational intersections between the themes.
    Summarize this into a 'Global Context Map' in under 500 words.
    Document content: {text[:15000]}...
    """
    response = await client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={"num_ctx": NUM_CTX, "temperature": 0.1}
    )
    return response['response']

async def evaluate_quality_with_judge(instruction: str, output: str, source_text: str) -> int:
    """Phase 4: LLM-as-a-Judge to filter out hallucinations."""
    prompt = f"""
    As a strict AI Auditor, review the following question and answer pair.
    Ensure it is entirely based on the provided source text and contains no fabricated information.
    Give a score from 1 to 5 (5 = perfectly faithful to the source, 1 = contains hallucinations).
    Strictly return ONLY JSON format: {{"score": number}}.

    Source Text: {source_text}
    Generated Question: {instruction}
    Generated Answer: {output}
    """
    try:
        response = await client.generate(
            model=OLLAMA_MODEL, prompt=prompt, format="json",
            options={"temperature": 0.0, "num_predict": 100}
        )
        score_data = JudgeScore.model_validate_json(extract_json(response['response']))
        return score_data.score
    except Exception as e:
        logging.error(f"Judge evaluation failed: {e}")
        return 0

async def process_chunk(chunk: str, global_context: str, quota: int, sem: asyncio.Semaphore, progress, task_id, global_state: dict, active_persona: str) -> int:
    """Phases 3 & 4: Process a single chunk asynchronously with strict state preservation."""
    chunk_hash = get_chunk_hash(chunk)
    
    if global_state.get(chunk_hash) == "completed":
        progress.update(task_id, advance=1)
        return 0

    async with sem:
        SchemaModel = get_schema()
        
        system_prompt = f"【Global Context】:\n{global_context}\n\n"
        format_req = (
            'must contain "instruction", "reasoning", and "output" keys' 
            if GENERATION_STYLE == 'cot' else 'must contain "instruction" and "output" keys'
        )
        
        base_prompt = f"""
        {system_prompt}
        Based on the following 【Current Chunk】, and referencing the depth of the 【Global Context】, generate {quota} training data pairs.
        All output text MUST be in {TARGET_LANGUAGE}.
        Strictly return a JSON Array format, where each object {format_req}. Do not output any other text.
        【Current Chunk】:\n{chunk}
        """

        retries = 3
        raw_json = ""
        current_prompt = base_prompt
        chunk_processed_successfully = False
        valid_data = []
        
        while retries > 0:
            valid_data = []  # Wipes clean on retry to prevent dirty data duplication
            try:
                response = await client.generate(
                    model=OLLAMA_MODEL, prompt=current_prompt, format="json",
                    options={"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "temperature": TEMPERATURE}
                )
                
                raw_json = extract_json(response['response'])
                parsed_list = json.loads(raw_json)
                
                if isinstance(parsed_list, dict):
                    parsed_list = [parsed_list]
                
                for item in parsed_list:
                    validated_item = SchemaModel(**item)
                    
                    # Low-Spec Hardware Optimization: Bypass AI Judge if threshold is set to minimum (1)
                    if MIN_QUALITY_SCORE <= 1:
                        score = 5
                    else:
                        score = await evaluate_quality_with_judge(
                            validated_item.instruction, validated_item.output, chunk
                        )
                    
                    if score >= MIN_QUALITY_SCORE:
                        valid_data.append(validated_item)
                    else:
                        logging.info(f"AI Judge discarded low-quality data (Score: {score})")
                
                chunk_processed_successfully = True
                break 
                
            except (json.JSONDecodeError, ValidationError) as e:
                logging.error(f"JSON parsing error. Retrying... Attempts left: {retries - 1}")
                # Clean retry prompt structure designed for easily distracted small models
                current_prompt = base_prompt + f"\n\n[CRITICAL ERROR CORRECTION]\nYour previous output failed validation with error: {str(e)}.\nDo not repeat the error. Rewrite the output as a pure JSON array now."
                retries -= 1
            except Exception as e:
                logging.error(f"Ollama API or Network error: {e}. Retrying... Attempts left: {retries - 1}")
                await asyncio.sleep(2)
                retries -= 1

        if chunk_processed_successfully:
            async with file_lock:
                out_path = OUTPUT_DIR / f"dataset_{GENERATION_STYLE}.jsonl"
                with open(out_path, 'a', encoding='utf-8') as f:
                    for item in valid_data:
                        content = (f"<think>\n{item.reasoning}\n</think>\n\n{item.output}" 
                                   if GENERATION_STYLE == 'cot' else item.output)
                        
                        sharegpt_record = {
                            "messages": [
                                {"role": "system", "content": active_persona},
                                {"role": "user", "content": item.instruction},
                                {"role": "assistant", "content": content}
                            ]
                        }
                        f.write(json.dumps(sharegpt_record, ensure_ascii=False) + '\n')

                global_state[chunk_hash] = "completed"
                save_checkpoint(global_state)
            
            progress.update(task_id, advance=1)
            return len(valid_data)
        else:
            logging.warning(f"Chunk abandoned after 3 failed attempts. Leaving uncompleted for next run.")
            progress.update(task_id, advance=1)
            return 0

# ==========================================
# 4. Main Execution Pipeline
# ==========================================
async def main():
    console.rule("[bold blue]Enterprise SLM Fine-Tuning Data Generation Pipeline Initialized[/bold blue]")
    
    raw_text = extract_and_clean_text()
    if not raw_text.strip():
        console.print("[red]Error: No valid text extracted. Ensure PDF files are in the data_input/ directory.[/red]")
        return
        
    if SYSTEM_PROMPT.lower() == "auto":
        active_persona = await detect_domain_persona(raw_text)
        console.print(f"🤖 [bold green]Auto-detected Persona:[/bold green] [magenta]{active_persona}[/magenta]")
    else:
        active_persona = SYSTEM_PROMPT

    encoder = tiktoken.get_encoding("cl100k_base")
    total_tokens = len(encoder.encode(raw_text, disallowed_special=()))
    max_safe_yield = total_tokens // TOKENS_PER_PAIR
    
    console.print(f"📄 Total Source Tokens: ~{total_tokens}")
    console.print(f"🛡️  Safe Generation Upper Limit ({GENERATION_STYLE.upper()}): [green]{max_safe_yield}[/green] pairs")

    if TARGET_YIELD > max_safe_yield:
        console.print(f"[bold red]Warning: Requested yield ({TARGET_YIELD}) exceeds the safe limit ({max_safe_yield})![/bold red]")
        if not AUTO_BYPASS:
            if not Confirm.ask("Do you wish to force execution anyway?"):
                console.print("[yellow]Execution aborted safely.[/yellow]")
                return

    global_context = await generate_global_context(raw_text)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks = splitter.split_text(raw_text)
    
    if TARGET_YIELD <= len(chunks):
        console.print(f"[yellow]Notice: TARGET_YIELD ({TARGET_YIELD}) is smaller than or equal to chunk count ({len(chunks)}). Using exactly {TARGET_YIELD} chunks.[/yellow]")
        chunks = chunks[:TARGET_YIELD]
        quotas = [1] * TARGET_YIELD
    else:
        base_quota = TARGET_YIELD // len(chunks)
        remainder = TARGET_YIELD % len(chunks)
        quotas = [base_quota + 1 if i < remainder else base_quota for i in range(len(chunks))]
    
    console.print(f"📦 Processing {len(chunks)} chunks to target exactly {TARGET_YIELD} pairs.")
    console.print("[dim](Note: Final saved yield may be slightly lower due to strict AI Judge filtering)[/dim]")

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    total_valid_yield = 0
    
    global_state = load_checkpoints()
    out_path = OUTPUT_DIR / f"dataset_{GENERATION_STYLE}.jsonl"
    
    if not global_state and out_path.exists():
        backup_path = OUTPUT_DIR / f"dataset_{GENERATION_STYLE}_backup_{int(time.time())}.jsonl"
        out_path.rename(backup_path)
        console.print(f"[yellow]⚠️ Notice: Empty state detected but previous output exists. Renamed old data to {backup_path.name} to prevent duplication.[/yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[cyan]Processing chunks and generating data...", total=len(chunks))
        
        tasks = [
            process_chunk(chunk, global_context, quotas[i], sem, progress, task_id, global_state, active_persona)
            for i, chunk in enumerate(chunks)
        ]
        
        results = await asyncio.gather(*tasks)
        total_valid_yield = sum(results)

    console.rule("[bold green]Pipeline Execution Complete[/bold green]")
    console.print(f"🎉 Successfully generated and saved [bold yellow]{total_valid_yield}[/bold yellow] high-quality {GENERATION_STYLE.upper()} data pairs!")
    console.print(f"📂 Output location: {OUTPUT_DIR}/dataset_{GENERATION_STYLE}.jsonl")

if __name__ == "__main__":
    asyncio.run(main())