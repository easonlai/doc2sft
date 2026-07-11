import asyncio
import os
import re
import json
import queue
import hashlib
import logging
import logging.handlers
import contextvars
import time
from datetime import datetime, timezone
from typing import List, Optional

# Load environment configuration before initializing pipeline
from dotenv import load_dotenv
load_dotenv()

from pypdf import PdfReader
from pydantic import BaseModel, Field, ValidationError
from ollama import AsyncClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =====================================================================
# 1. OBSERVABILITY & TELEMETRY ENGINE (COROUTINE-SAFE)
# =====================================================================
logger_ctx = contextvars.ContextVar("logger_ctx", default={})

class ContextInjectingFilter(logging.Filter):
    """Injects async context variables into the LogRecord BEFORE it hits the background thread."""
    def filter(self, record):
        ctx = logger_ctx.get()
        # Attach variables directly to the record so all formatters can see them
        record.pipeline_stage = getattr(record, "stage", ctx.get("stage", "system_execution"))
        record.trace_id = getattr(record, "trace_id", ctx.get("trace_id", "N/A"))
        record.file_name = getattr(record, "file_name", ctx.get("file_name", "N/A"))
        record.chunk_index = getattr(record, "chunk_index", ctx.get("chunk_index", None))
        return True

class ProductionJSONFormatter(logging.Formatter):
    """Bulletproof, non-blocking JSON formatter for high-concurrency async pipelines."""
    def format(self, record):
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "pipeline_stage": record.pipeline_stage,
            "trace_id": record.trace_id,
            "file_name": record.file_name,
            "chunk_index": record.chunk_index,
            "message": record.getMessage()
        }
        
        telemetry_keys = [
            "retry_count", "ai_score", "error_type", 
            "latency_ms", "raw_output_length", "pydantic_errors"
        ]
        
        for key in telemetry_keys:
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)
                
        if record.exc_info:
            log_obj["stack_trace"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj, default=str)

# Initialize Master Logger
logger = logging.getLogger("Doc2SFT_Telemetry")
logger.setLevel(logging.DEBUG)

# Attach the filter to the logger so it runs in the main async thread
logger.addFilter(ContextInjectingFilter())

# Initialize Non-Blocking Background Queue Logger
log_queue = queue.Queue(-1)
queue_handler = logging.handlers.QueueHandler(log_queue)
logger.addHandler(queue_handler)

os.makedirs("logs", exist_ok=True)
file_target = logging.FileHandler("logs/traceability.jsonl", encoding="utf-8")
file_target.setFormatter(ProductionJSONFormatter())
file_target.setLevel(logging.DEBUG)

console_target = logging.StreamHandler()
console_target.setFormatter(logging.Formatter('%(levelname)s - [%(pipeline_stage)s] %(message)s'))
console_target.setLevel(logging.INFO)

listener = logging.handlers.QueueListener(log_queue, file_target, console_target, respect_handler_level=True)
listener.start()

# =====================================================================
# 2. CONFIGURATION LOADER
# =====================================================================
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
INPUT_DIR = os.getenv("INPUT_DIR", "./data_input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./data_output")
CHECKPOINT_FILE = os.getenv("CHECKPOINT_FILE", "./logs/state.json")
GENERATION_STYLE = os.getenv("GENERATION_STYLE", "qa")

# Performance & Context Window Management
MIN_QUALITY_SCORE = int(os.getenv("MIN_QUALITY_SCORE", "4"))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "2"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX_QA", "4096"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT_QA", "1000"))

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Shared Global Locks for Async File Access
file_write_lock = asyncio.Lock()
state_write_lock = asyncio.Lock()

# =====================================================================
# 3. DATA SCHEMA SCHEMATICS (PYDANTIC V2)
# =====================================================================
class ShareGPTMessage(BaseModel):
    role: str = Field(..., description="System, user, or assistant role")
    content: str = Field(..., description="The main text body")

class ShareGPTData(BaseModel):
    messages: List[ShareGPTMessage]

class JudgeScore(BaseModel):
    score: int = Field(..., ge=1, le=5, description="Factual matching score 1 to 5")
    reasoning: str = Field(..., description="Audit justification")

# =====================================================================
# 4. UTILITIES & DATA PARSING EXTRACTIONS
# =====================================================================
def omni_extract_json(raw_string: str) -> str:
    """Uses robust boundary detection to safely extract hidden JSON structures."""
    raw_string = raw_string.strip()
    
    start_obj, end_obj = raw_string.find('{'), raw_string.rfind('}')
    start_arr, end_arr = raw_string.find('['), raw_string.rfind(']')
    
    if start_obj != -1 and end_obj != -1 and (start_arr == -1 or start_obj < start_arr):
        return raw_string[start_obj:end_obj+1]
    elif start_arr != -1 and end_arr != -1:
        return raw_string[start_arr:end_arr+1]
        
    return raw_string

async def call_ollama_endpoint(prompt: str, system: Optional[str] = None, response_format: Optional[str] = None) -> str:
    """Handles async inference using the official Ollama AsyncClient with an optional JSON Grammar Mask."""
    client = AsyncClient(host=OLLAMA_HOST)
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT
        }
    }
    
    # Strictly enforce C-level JSON grammar rendering if requested
    if response_format == "json":
        payload["format"] = "json"
        
    if system:
        payload["system"] = system

    try:
        response = await client.generate(**payload)
        return response.get("response", "")
    except Exception as e:
        raise RuntimeError(f"Ollama Endpoint Failed: {str(e)}")

# =====================================================================
# 5. DATA INGESTION & SEMANTIC SEGMENTATION
# =====================================================================
def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    return "".join([page.extract_text() or "" for page in reader.pages])

async def generate_global_context_map(full_text: str, file_name: str) -> str:
    """Creates a high-level metadata layout of the entire document to combat chunk myopia."""
    logger_ctx.set({"stage": "ingestion_context_mapping", "file_name": file_name, "trace_id": "global-map"})
    logger.info(f"Initiating Global Context Map sequence for {file_name}.")
    
    truncated_text = full_text[:12000]
    prompt = (
        f"Analyze this document snippet and generate a dense 3-sentence summary highlighting the primary "
        f"domain, core objective, and technical context. Output ONLY the summary.\n\nText:\n{truncated_text}"
    )
    try:
        start_time = time.perf_counter()
        context_map = await call_ollama_endpoint(prompt) # Purposefully omits JSON format
        latency = int((time.perf_counter() - start_time) * 1000)
        logger.info("Global Context Map locked and compiled.", extra={"latency_ms": latency})
        return context_map
    except Exception as e:
        logger.error(f"Failed to generate Global Context Map: {str(e)}", exc_info=True)
        return "General enterprise operational context guidelines."

def recursive_chunk_engine(text: str, size: int = 2000, overlap: int = 300) -> List[str]:
    """Uses LangChain to ensure chunks break cleanly at paragraph or sentence boundaries."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_text(text)

# =====================================================================
# 6. CORE PROCESSING PIPELINE STATE MACHINE
# =====================================================================
async def process_chunk(chunk_text: str, chunk_idx: int, file_name: str, file_hash: str, global_map: str):
    trace_id = f"{file_hash}-chunk-{chunk_idx}"
    logger_ctx.set({"trace_id": trace_id, "file_name": file_name, "chunk_index": chunk_idx, "stage": "async_dispatch"})
    
    logger.debug(f"Chunk {chunk_idx} dispatched to pipeline execution slot.")
    
    # -----------------------------------------------------------------
    # STAGE 1 & 2: GENERATION WITH SELF-HEALING RETRY LOOPS
    # -----------------------------------------------------------------
    max_retries = 3
    retry_count = 0
    validated_data: Optional[ShareGPTData] = None
    
    base_prompt = (
        f"You are a training data generation engine. Based on the following context, generate a JSON array "
        f"containing 2 high-quality Question/Answer pairs matching the ShareGPT standard format schema. "
        f"Ensure values are factual and directly inferred from the chunk text.\n\n"
        f"Global System Context: {global_map}\n\n"
        f"Chunk Ground Truth Text:\n{chunk_text}\n\n"
        f"OUTPUT ONLY RAW JSON COMPLYING EXACTLY WITH THIS PYDANTIC EXPECTATION:\n"
        f'{{"messages": [{{"role": "user", "content": "question"}}, {{"role": "assistant", "content": "answer"}}]}}'
    )
    
    current_prompt = base_prompt
    
    while retry_count < max_retries:
        try:
            ctx = logger_ctx.get().copy()
            ctx["stage"] = "llm_inference"
            logger_ctx.set(ctx)
            
            start_time = time.perf_counter()
            # Enforce JSON Mask
            raw_output = await call_ollama_endpoint(current_prompt, response_format="json")
            latency = int((time.perf_counter() - start_time) * 1000)
            
            ctx["stage"] = "pydantic_validation"
            logger_ctx.set(ctx)
            
            clean_json = omni_extract_json(raw_output)
            
            if not clean_json.startswith("{") and not clean_json.startswith("["):
                raise ValueError("Extracted string failed structural bounding constraints.")
                
            if clean_json.startswith("["):
                clean_json = f'{{"messages": {clean_json}}}'
                
            validated_data = ShareGPTData.model_validate_json(clean_json)
            logger.debug("Structural validation passed.", extra={"latency_ms": latency, "raw_output_length": len(raw_output)})
            break
            
        except (ValidationError, ValueError, Exception) as e:
            retry_count += 1
            ctx = logger_ctx.get().copy()
            ctx["stage"] = "retry_engine"
            logger_ctx.set(ctx)
            
            pydantic_payload = e.errors() if isinstance(e, ValidationError) else [{"msg": str(e)}]
            
            logger.warning(
                f"Validation collapsed. Initiating isolated retry attempt {retry_count}/{max_retries}.",
                extra={"retry_count": retry_count, "error_type": type(e).__name__, "pydantic_errors": pydantic_payload}
            )
            current_prompt = f"{base_prompt}\n\nCRITICAL FIX REQUIRED: Your previous execution failed with structural error: {str(pydantic_payload)}. Re-output clean valid schema elements."
            
    if not validated_data:
        ctx = logger_ctx.get().copy()
        ctx["stage"] = "quarantine"
        logger_ctx.set(ctx)
        logger.error(f"Max system processing loops exhausted for Chunk {chunk_idx}. Data quarantined.", extra={"retry_count": retry_count})
        return
        
    # -----------------------------------------------------------------
    # STAGE 3: ENTERPRISE AUDITOR (LLM-AS-A-JUDGE HALLUCINATION FILTER)
    # -----------------------------------------------------------------
    if MIN_QUALITY_SCORE > 1:
        ctx = logger_ctx.get().copy()
        ctx["stage"] = "ai_judge_audit"
        logger_ctx.set(ctx)
        
        judge_prompt = (
            f"Act as a ruthless enterprise Data Compliance Auditor. Evaluate if the generated QA metrics "
            f"accurately reflect the ground truth content without introducing hallucinations or outside assumptions.\n\n"
            f"Ground Truth Slice:\n{chunk_text}\n\n"
            f"Generated Pipeline Output:\n{validated_data.model_dump_json()}\n\n"
            f"Return a JSON object specifying a quality score integer from 1 to 5 (5 being pristine) and a clear reasoning string.\n"
            f"OUTPUT ONLY VALUED FORMAT MATCHING:\n"
            f'{{"score": 5, "reasoning": "text"}}'
        )
        
        try:
            # Enforce JSON Mask for the auditor as well
            judge_raw = await call_ollama_endpoint(judge_prompt, response_format="json")
            judge_clean = omni_extract_json(judge_raw)
            parsed_score = JudgeScore.model_validate_json(judge_clean)
            
            logger.info(
                f"AI Judge audit completed. Score: {parsed_score.score}/5", 
                extra={"ai_score": parsed_score.score, "judge_reasoning": parsed_score.reasoning}
            )
            
            if parsed_score.score < MIN_QUALITY_SCORE:
                logger.warning(f"Chunk {chunk_idx} discarded due to low quality score assignment.")
                return
        except Exception as e:
            logger.error(f"Auditor routing collapsed: {str(e)}. Defaulting to rejection safety protocol.")
            return

    # -----------------------------------------------------------------
    # STAGE 4: ASYNC ATOMIC WRITES & STATE COMMIT
    # -----------------------------------------------------------------
    ctx = logger_ctx.get().copy()
    ctx["stage"] = "state_commit"
    logger_ctx.set(ctx)
    
    async with file_write_lock:
        output_path = os.path.join(OUTPUT_DIR, f"dataset_{GENERATION_STYLE}.jsonl")
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(validated_data.model_dump_json() + "\n")
            
    async with state_write_lock:
        state_data = {}
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, "r", encoding="utf-8") as sf:
                    state_data = json.load(sf)
            except Exception:
                state_data = {}
                
        if file_hash not in state_data:
            state_data[file_hash] = []
        state_data[file_hash].append(chunk_idx)
        
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as sf:
            json.dump(state_data, sf, indent=2)
            
    logger.info(f"Pipeline processing completely closed out for Chunk {chunk_idx}. Chkpnt written.", extra={"status": "completed"})

# =====================================================================
# 7. ASYNC ORCHESTRATION & SEMAPHORE CONTROL HANDLERS
# =====================================================================
async def main_orchestration_loop():
    logger_ctx.set({"stage": "system_initialization", "trace_id": "core-init"})
    logger.info("Initializing Doc2SFT Master Core Orchestration Pipeline Engine.")
    
    checkpoint_state = {}
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as sf:
                checkpoint_state = json.load(sf)
        except Exception:
            logger.warning("State checkpoint file unreadable. Overwriting with clean blueprint framework mappings.")

    target_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    if not target_files:
        logger.error(f"Zero source documents located in '{INPUT_DIR}'. Execution halted.")
        return

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async def bounded_worker(chunk_text, chunk_idx, file_name, file_hash, global_map):
        async with semaphore:
            await process_chunk(chunk_text, chunk_idx, file_name, file_hash, global_map)

    tasks = []
    for file_name in target_files:
        file_path = os.path.join(INPUT_DIR, file_name)
        
        with open(file_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            
        logger.info(f"Ingesting: '{file_name}' | Fingerprint Map Identity: [{file_hash}]")
        
        try:
            raw_text = extract_pdf_text(file_path)
            if not raw_text.strip():
                logger.warning(f"File '{file_name}' possesses no printable text objects. Skipping.")
                continue
                
            global_map = await generate_global_context_map(raw_text, file_name)
            chunks = recursive_chunk_engine(raw_text)
            completed_indices = checkpoint_state.get(file_hash, [])
            
            for idx, chunk_content in enumerate(chunks):
                if idx in completed_indices:
                    continue
                tasks.append(bounded_worker(chunk_content, idx, file_name, file_hash, global_map))
                
        except Exception as e:
            logger.critical(f"Inability to initialize structural parsing constraints against '{file_name}': {str(e)}", exc_info=True)

    if tasks:
        logger.info(f"Asynchronous queue compiled. Spawning {len(tasks)} isolated context validation workers.")
        await asyncio.gather(*tasks)
    else:
        logger.info("All detected storage block components match active checkpoint indexes. Zero execution steps required.")

    logger.info("Pipeline execution operations finalized. Tearing down background telemetry logging loops safely.")
    listener.stop()

if __name__ == "__main__":
    asyncio.run(main_orchestration_loop())