# MediChar

MediChar is a local-first ICD-10 workbench built around a FastAPI backend and a Streamlit frontend. It combines:

- ICD-10 retrieval over the MIMIC-IV demo diagnosis dictionary
- diagnosis-centered candidate generation and safety filtering
- LLM-based prioritization after retrieval
- direct ICD code lookup
- printed and handwritten note OCR with review-before-predict flow

The current product flow is:

`note text or image -> OCR review if needed -> /predict -> ranked ICD-10 output`

Image OCR is extraction-only in the frontend. The user reviews or edits extracted text before sending it to prediction.

## Features

- `Disease to Code`: paste a clinical note and get ranked ICD-10 suggestions
- `Code to Disease`: look up one or more ICD codes directly
- `Printed OCR`: Windows OCR for typed/scanned notes
- `Handwritten OCR`: local TrOCR path with handwritten-focused cleanup
- `Review tabs`: final OCR text, raw OCR text, cleaned OCR text, and uncertain fragments
- `Grounded output`: summary, condition overview, precautions, health guidance, retrieved context, and prioritization reasoning

## Project Layout

```text
MediChar/
|-- frontend/
|   `-- app.py
|-- backend/
|   |-- api/
|   |   |-- routes/
|   |   `-- schemas/
|   |-- core/
|   |-- data/
|   |   |-- mimic-iv-clinical-database-demo-2.2/
|   |   |-- ocr_benchmark_cases.json
|   |   `-- processed/
|   |-- scripts/
|   |-- services/
|   |-- tests/
|   `-- main.py
|-- requirements.txt
`-- README.md
```

## Run Locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python backend\scripts\build_vector_index.py
uvicorn backend.main:app --reload
streamlit run frontend\app.py
```

If you do not want CUDA acceleration, install a CPU build of PyTorch instead. The handwritten OCR path will still run, but slower.

## Environment

The backend loads configuration from `backend/.env`.

Important variables:

- `MEDICHAR_LLM_PROVIDER=searchapi|heuristic|auto`
- `MEDICHAR_LLM_MODEL=google_ai_mode`
- `MEDICHAR_EMBEDDING_MODEL=all-MiniLM-L6-v2`
- `MEDICHAR_RETRIEVAL_TOP_K=5`
- `MEDICHAR_RETRIEVAL_MIN_SCORE=0.2`
- `MEDICHAR_REQUEST_TIMEOUT=60`
- `MEDICHAR_VISION_PROVIDER=local|openai|disabled`
- `MEDICHAR_VISION_MODEL=microsoft/trocr-base-handwritten`
- `MEDICHAR_VISION_DEVICE=auto|cuda|cpu`
- `MEDICHAR_VISION_TIMEOUT=60`
- `MEDICHAR_OCR_PREPROCESSING_ENABLED=true|false`
- `MEDICHAR_OCR_PREPROCESSING_MODE=always|auto|off`
- `MEDICHAR_API_BASE_URL=http://localhost:8000`
- `SEARCHAPI_API_KEY`
- `SEARCHAPI_BASE_URL`
- `SEARCHAPI_ENGINE`
- `SEARCHAPI_HL`
- `SEARCHAPI_GL`
- `OPENAI_API_KEY`

## API Summary

- `POST /predict`
  Accepts clinical text and returns ICD-10 suggestions.

- `POST /extract-note-image`
  Accepts an image and returns:
  - final extracted text
  - raw OCR text
  - cleaned OCR text
  - confidence
  - uncertain fragments

- `POST /predict-image`
  Kept for compatibility. The frontend does not use this path.

- `GET /lookup/{code}`
  Returns title, definition, hints, and suggestions for a single ICD code.

- `POST /lookup-batch`
  Returns lookup results for multiple ICD codes.

- `GET /health`
  Returns service readiness plus model/provider metadata.

## Frontend Workflow

### Disease to Code

1. Paste a note, load a curated dataset-based example, or upload an image.
2. If using an image, choose `Printed` or `Handwritten`.
3. Run `Extract Text From Image`.
4. Review the OCR output in:
   - `Editable Final Text`
   - `Raw OCR`
   - `Cleaned OCR`
5. Replace or append the reviewed OCR text into the main note box.
6. Run `Analyze Note`.

### Code to Disease

1. Enter one or more ICD codes.
2. Run `Lookup Code`.
3. Review the grounded title, definition, hint, and suggestions.

## Example Notes

The frontend now uses longer curated examples based on dataset ICD-10 titles, not one-line placeholder text. The preferred demo examples were selected to stay stable under the current model:

- `A419` Sepsis, unspecified organism
- `I10` Essential (primary) hypertension
- `J189` Pneumonia, unspecified organism
- `N390` Urinary tract infection, site not specified

These examples were checked against the local prediction path before being kept as defaults.

## OCR Benchmarking

Use the benchmark scaffold to evaluate OCR changes consistently:

```powershell
python backend\scripts\evaluate_ocr_pipeline.py
```

Before running it, update:

- [ocr_benchmark_cases.json](/C:/Users/Parth/Desktop/Projects/MediChar/backend/data/ocr_benchmark_cases.json)

Each case should point to a local image and expected transcription. The script reports word-overlap, line-recall, uncertain-fragment counts, and predicted text.

## Tests

```powershell
python -m unittest discover -s backend\tests
python -m compileall backend frontend
```

Useful focused checks:

```powershell
python -m unittest backend.tests.test_ocr_image_modes
python -m compileall backend\scripts\evaluate_ocr_pipeline.py
```

## Current Technical Limits

- Printed OCR is stronger than handwritten OCR.
- Handwritten OCR is still limited by document perception and line grouping.
- The ICD engine is strongest when the note includes explicit assessment/diagnosis language.
- The OCR benchmark file is a scaffold and still needs real local evaluation cases.




## Detailed Backend Working

This section explains the project end to end in detail so the internal flow

### 1. Overall System Idea

MediChar is not a single-model application. It is a pipeline made of:

- dataset-backed ICD dictionary loading
- text normalization and vocabulary support
- embedding generation
- FAISS vector retrieval
- diagnosis-centered candidate expansion
- generic safety filtering
- semantic reranking
- LLM-based prioritization
- OCR extraction for printed and handwritten notes
- frontend review before final prediction

The project is local-first in the sense that the ICD knowledge base, vector index, OCR extraction, and application logic run locally. The LLM layer is configurable. If SearchAPI is available, the backend uses it for generation and prioritization. If not, the backend falls back to a local heuristic mode so the application still works.

### 2. Dataset and Knowledge Base

The project does not train a medical coding model from scratch. Instead, it uses the ICD-10 diagnosis dictionary from the MIMIC-IV Clinical Database Demo dataset as a local knowledge base.

Active source file:

- `backend/data/mimic-iv-clinical-database-demo-2.2/hosp/d_icd_diagnoses.csv`

What this file provides:

- ICD code
- ICD version
- long diagnosis title

This means the system is not learning from full physician notes in MIMIC. It is using the ICD catalog titles as the retrieval corpus. That is why retrieval quality depends heavily on semantic matching, diagnosis extraction, and filtering logic.

### 3. Startup and Configuration

The backend entrypoint is:

- `backend/main.py`

At startup the application:

1. reads environment variables from `backend/.env`
2. creates a `RAGSettings` object from `backend/core/config.py`
3. ensures required folders exist
4. initializes `MedicalCodingService`

Main settings include:

- embedding model name
- retrieval thresholds
- LLM provider and timeout
- OCR provider and OCR model
- OCR cleanup thresholds
- semantic reranking weights

This design keeps runtime behavior configurable without changing source code.

### 4. Main Libraries Used

Core backend and frontend libraries:

- `fastapi`
  Used for API routes, request validation, dependency injection, and response models.

- `uvicorn`
  ASGI server used to run the FastAPI backend locally.

- `streamlit`
  Frontend framework used for the interactive workbench UI.

- `sentence-transformers`
  Used to load the embedding model `all-MiniLM-L6-v2` and convert ICD titles and user text into vector embeddings.

- `faiss-cpu`
  Used for similarity search over the ICD vector index.

- `numpy`
  Used for embedding arrays and OCR image processing support.

- `python-dotenv`
  Loads backend environment variables from `backend/.env`.

- `requests`
  Used for calling the LLM provider and for frontend-to-backend API requests.

- `rapidfuzz`
  Used inside OCR cleanup for fuzzy token correction.

- `opencv-python`
  Used for local handwritten OCR preprocessing, thresholding, line detection, and notebook-line suppression.

- `torch`
  Used to run the handwritten OCR model locally.

- `transformers`
  Used to load TrOCR for handwritten note extraction.

- `python-multipart`
  Required for image upload routes in FastAPI.

Other components:

- Windows OCR through the local WinRT-based path for printed text
- optional OpenAI vision client exists in code, but the final intended flow is local OCR first

### 5. ICD Data Loading and Index Creation

The ICD catalog is loaded through:

- `backend/services/data_loader.py`

The vector index is created through:

- `backend/scripts/build_vector_index.py`

How this works:

1. the ICD CSV is read
2. only ICD-10 rows are kept
3. each row is converted into a structured entry containing:
   - code
   - title
   - keywords
   - definition
   - explanation hint
   - chunk text
4. all chunk texts are embedded using `SentenceTransformerEmbedder`
5. embeddings are stored in a FAISS inner-product index
6. metadata records are saved alongside the index

Relevant implementation:

- `backend/services/embedding.py`
- `backend/services/vector_store.py`
- `backend/services/retriever.py`

Why FAISS is used:

- retrieval is fast
- embeddings are normalized
- similarity search is efficient even when the ICD catalog is large

### 6. What Happens In `/predict`

The main API route is:

- `POST /predict`

The request goes through:

- `backend/api/routes/predict.py`

That route calls:

- `MedicalCodingService.predict(...)`

The internal text pipeline is roughly:

`clinical text -> diagnosis context -> retrieval -> candidate augmentation -> safety filtering -> semantic reranking -> code generation -> LLM prioritization -> validation -> final response`

#### Step 6.1: Raw Text Intake

The request body contains:

- note text
- `top_k`
- whether to return retrieved context
- whether to return similarity scores

The backend does not assume the input is perfect. It can still run OCR-style cleanup on text if it looks noisy enough, but this was tightened so that clean typed notes do not take the OCR cleanup path unnecessarily.

#### Step 6.2: Diagnosis-Centered Context Building

Instead of treating the entire note as one bag of words, the system tries to extract diagnosis-focused phrases first.

Examples:

- `sepsis secondary to pneumonia`
- `acute kidney injury`
- `essential hypertension`

This is done by looking at diagnosis-relevant sections and splitting on connectors like:

- `with`
- `and`
- `secondary to`
- `due to`
- `complicated by`

This is important because a note may contain more than one clinically valid diagnosis, and retrieval should not stop after only one match.

#### Step 6.3: Multi-Query Retrieval

The retriever does not search only once.

It uses:

- the whole note
- extracted diagnosis fragments

The whole note can still use richer focus-query expansion. Fragment queries use a leaner retrieval path to reduce latency.

This behavior is implemented in:

- `backend/services/retriever.py`
- `backend/services/medical_coding.py`

The retrieved records include:

- code
- title
- keywords
- definition
- hint
- chunk text
- similarity score

#### Step 6.4: Catalog Match Augmentation

After retrieval, the backend can add extra catalog-supported candidates if the note contains a diagnosis phrase that strongly overlaps an ICD title but retrieval did not rank it high enough initially.

This helps in cases like:

- explicit pneumonia mention being more important than a generic lower-respiratory similarity hit
- explicit AKI mention being stronger than a loosely related kidney title

#### Step 6.5: Generic Safety Filtering

The system then removes or demotes unsafe candidates without hardcoding fixed disease-to-code mappings.

Examples of filtering rules:

- no `S` or `T` injury code without trauma context
- no pregnancy `O` code without pregnancy context
- no newborn/perinatal `P` code without newborn context
- no complication-heavy diabetes code if those complications are not supported

This is one of the main protections against obviously wrong but semantically similar matches.

#### Step 6.6: Semantic Reranking

The project includes a semantic feature extraction layer and candidate alignment layer:

- `backend/services/semantic_feature_extractor.py`
- `backend/services/feature_alignment.py`
- `backend/services/reranker.py`

The note is converted into structured signals such as:

- anatomy
- severity
- temporal descriptors
- diagnostics
- qualifiers
- symptom clusters

Each retrieved ICD candidate is then aligned against those features and given a combined score based on:

- similarity score
- feature alignment score
- specificity/detail overlap

This is how the project moves beyond plain vector similarity.

#### Step 6.7: Code Generation

Once the candidate set is cleaned and reranked, the backend calls the LLM generation layer.

Two modes exist:

- `SearchApiLLMClient`
  Uses SearchAPI for JSON-formatted code generation and code prioritization.

- `HeuristicLLMClient`
  Local fallback when external LLM use is unavailable.

The generation prompt includes:

- retrieved ICD context
- full uploaded clinical note
- extracted note evidence
- structured semantic signals

The LLM is instructed to:

- extract all supported diagnoses
- avoid unsupported codes
- keep explanations case-aware
- avoid symptom-only replacement when a diagnosis is documented

#### Step 6.8: LLM Prioritization

After code generation, the backend performs a second reasoning step:

- which code is primary
- which codes are secondary
- which retrieved candidates should be dropped

This is done only after retrieval. The LLM does not invent a fresh ICD universe from scratch. It prioritizes among candidates already grounded in the local catalog.

This step returns:

- `primary_icd`
- `secondary_icd`
- `dropped_codes`
- `prioritization_reasoning`

If only one code survives, the extra prioritization call is skipped to reduce latency.

#### Step 6.9: Final Validation and Response

Before the final response is returned, the backend validates selected predictions again against:

- the retrieved candidate set
- safety filters
- code normalization rules

The final JSON response can contain:

- ranked codes
- confidence labels
- summary
- condition overview
- precautions
- diet / health advice
- prioritization reasoning
- retrieved context

### 7. How `/lookup` Works

There is also a direct code-to-disease pipeline:

- `GET /lookup/{code}`
- `POST /lookup-batch`

This path does not use the note prediction pipeline. It normalizes ICD code input and checks the loaded catalog directly.

If an exact code is not found:

- suggestions are generated from normalized code prefixes and code similarity

This is useful during demo because it shows the catalog itself is accessible independently of note prediction.

### 8. OCR Flow

OCR is split into two explicit modes.

#### Printed OCR

Printed notes use:

- `WindowsOCRVisionClient`

This is better for typed or clean scanned documents.

Printed OCR flow:

1. image uploaded from frontend
2. backend route `/extract-note-image` receives multipart data
3. printed OCR client extracts text using the Windows OCR path
4. OCR cleanup can still run
5. final extracted text is returned to frontend for user review

#### Handwritten OCR

Handwritten notes use:

- `LocalTrOCRVisionClient`

This path is slower because it performs:

1. image decoding
2. OpenCV preprocessing
3. line/region detection
4. region cropping
5. TrOCR generation on each detected region
6. assembly into note text
7. conservative post-processing

##### Handwritten OCR preprocessing details

Inside `local_vision.py`, the handwritten image pipeline does:

- grayscale conversion
- Gaussian blur
- adaptive thresholding
- notebook-line suppression
- morphology and dilation
- row projection analysis
- line band extraction
- refined line region cropping
- resizing and sharpening before TrOCR

The project also now merges likely continuation lines so structures like:

- `HPI:`
- followed by indented continuation text

stay together better.

##### Handwritten OCR post-processing details

After TrOCR extraction, the result goes through:

- cleaning
- fuzzy correction
- uncertainty detection
- optional reconstruction

This is handled by:

- `backend/services/ocr_preprocessor.py`

Important design choice:

- handwritten OCR no longer always forces aggressive reconstruction
- if raw TrOCR text is cleaner than the processed version, the raw version is kept
- common shorthand like `SOB`, `HPI`, `Dx`, `O2`, `RLL`, `CXR`, `IV` is protected

### 9. Why OCR Is Extraction-Only In Frontend

The frontend intentionally does not send image OCR directly into prediction without user review.

Actual user flow:

1. upload image
2. choose `Printed` or `Handwritten`
3. run `Extract Text From Image`
4. inspect:
   - final editable text
   - raw OCR
   - cleaned OCR
   - uncertain fragments
5. replace or append reviewed text into the main note box
6. click `Analyze Note`

This separation was intentional because OCR errors should not silently flow into ICD prediction.

### 10. Frontend Working

The frontend is a Streamlit application in:

- `frontend/app.py`

It provides two workflows.

#### Disease to Code

Capabilities:

- load curated long demo notes
- paste custom note text
- upload image and extract note text
- review OCR output
- analyze note
- see ranked ICD suggestions and retrieved context

#### Code to Disease

Capabilities:

- lookup one ICD code
- lookup multiple ICD codes
- show title, definition, hints, and suggestions

The frontend uses `requests` to call the backend API and keeps OCR extraction separate from note prediction.

### 11. Demo Examples In Frontend

The one-line placeholder examples were removed. The frontend now uses curated longer note examples chosen to be more realistic and stable under the current model.

Preferred demo examples:

- `A419`
- `I10`
- `J189`
- `N390`

These were intentionally checked against the local prediction path so they do not embarrass the demo by drifting to obviously wrong codes.

### 12. Benchmarking and Testing

The project includes:

- unit tests in `backend/tests`
- OCR benchmark scaffold in `backend/data/ocr_benchmark_cases.json`
- OCR evaluation script in `backend/scripts/evaluate_ocr_pipeline.py`

What the benchmark is meant to measure:

- word overlap
- line recall
- uncertain fragment count
- printed vs handwritten OCR behavior

This is useful because OCR tuning without a benchmark becomes subjective very quickly.

### 13. Important Engineering Tradeoffs

The project makes several explicit tradeoffs.

#### Retrieval vs direct generation

The system is retrieval-grounded because:

- ICD coding must stay linked to a local catalog
- raw LLM generation alone can hallucinate unsupported codes

#### Local OCR vs frontier multimodal vision

The project uses a local OCR stack because:

- it is controllable
- it is demonstrable offline/local-first
- it is cheaper to run repeatedly

But this also means handwritten quality will not match frontier multimodal systems like ChatGPT Vision.

#### Accuracy vs latency

The system uses:

- diagnosis-centered retrieval
- semantic reranking
- optional OCR cleanup
- optional LLM prioritization

These improve quality, but add time. Some latency work was already done by:

- reducing unnecessary OCR-style preprocessing for clean notes
- reducing query expansion on fragment retrieval
- skipping extra LLM prioritization for single-code outputs

### 14. Technical Strengths Of The Project

From an interview perspective, the strongest technical points are:

- modular FastAPI + Streamlit architecture
- local vector search with FAISS
- diagnosis-centered retrieval instead of only whole-note similarity
- generic safety filtering without hardcoded disease-specific ICD mappings
- OCR review-before-predict design
- multi-step reasoning:
  retrieval -> reranking -> prioritization -> validation
- graceful fallback from external LLM to heuristic mode

### 15. Known Limitations

Important to state honestly:

- the system uses ICD catalog titles, not a large labeled note-to-code training set
- printed OCR is stronger than handwritten OCR
- handwritten OCR remains limited by page perception and region grouping
- external LLM quality depends on provider availability and timeout behavior
- benchmark scaffolding exists, but should still be populated with a larger real evaluation set

### 16. One-Line Interview Summary

If you need a compact summary for interview:

`MediChar is a local-first ICD-10 coding workbench where clinical notes or OCR-extracted note text are grounded against the MIMIC-IV ICD dictionary using sentence-transformer embeddings and FAISS retrieval, then refined by diagnosis-centered filtering, semantic reranking, LLM prioritization, and validation before returning ranked ICD-10 suggestions with explanations and review context.`
