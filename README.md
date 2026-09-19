# Legal Metrology Compliance Checking System (Backend Prototype)

A complete, runnable backend prototype for an automated **Legal Metrology Regulatory Compliance Checking System** built with **FastAPI**, **SQLite**, **PaddleOCR**, **OpenCV**, **ReportLab**, **python-docx**, and **BeautifulSoup4**.

This system runs **100% locally with zero cloud dependencies, no paid APIs, and no external billing requirements**.

---

## Key Features

1. **Rule 6 Regulatory Compliance Engine**:
   - Automated verification of mandatory declarations under the **Legal Metrology Act, 2009** and the **Legal Metrology (Packaged Commodities) Rules, 2011**.
   - Verifies **MRP** (Maximum Retail Price), **Net Quantity**, **Date of Manufacture/Packaging**, **Manufacturer / Packer / Importer Details**, **Consumer Care / Contact Information**, and **Country of Origin**.
   - Handles commodity categories: `general_retail`, `food` (includes Best Before / Expiry), `cosmetics`, `electronics`, and `ecommerce_listing` (automatically excludes manufacturing date at the listing stage as per Legal Metrology e-commerce norms).
2. **3-Layer Field Matching Architecture**:
   - **Layer 1: Exact / Synonym Dictionary Matching**: Fast matching against a pre-compiled statutory synonym dictionary for all Rule 6 declarations (e.g. `mrp` matches "MRP", "Maximum Retail Price", "M.R.P.", "Retail Price", etc.).
   - **Layer 2: Fuzzy String Matching (`rapidfuzz`)**: Fallback for OCR typos, near-misses, and character substitutions (e.g. "Max Retall Prlce", "Mfe Date", "Nt Quantlty") with a similarity threshold of $\ge 85\%$.
   - **Layer 3: Google Gemini API (gemini-3.8-flash) Fallback**: When Layer 1 & 2 fail, passes ambiguous text snippets to the Gemini API to classify which standard declaration it corresponds to. It also uses Gemini Vision for low-confidence or distorted image regions.
   - **Separate AI Match Audit Logging**: Every match resolved by Layer 3 is logged to `data/ai_resolved_matches.jsonl` and accessible via `GET /dashboard/ai-matches` for admin review and promotion into the synonym dictionary.
3. **Quality & Preprocessing Pipeline**:
   - **Laplacian Variance Blur Detection**: Rejects blurry or out-of-focus label photographs with actionable feedback.
   - **Deskewing & Contrast Enhancement**: Aligns rotated text and applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on packaging surfaces.
4. **Local PaddleOCR Inference**:
   - Runs lightweight, high-speed mobile models locally on CPU.
   - Extracts line bounding boxes, polygon coordinates, confidence scores, and raw text.
5. **Font Size Anomaly Detection**:
   - Computes the median text height across all printed declarations on the label.
   - Flags required declarations that are significantly smaller than the median text size (`< 0.50 * median`), marking them for officer review.
6. **E-Commerce Listing Auditing**:
   - Scrapes product marketplace URLs using BeautifulSoup to audit digital declarations.
7. **Executive Inspection Reports**:
   - **PDF Reports** generated via ReportLab with formal government/statutory styling and audit tables.
   - **DOCX Reports** generated via python-docx for editing and filing.
8. **Officer Review Workflow & Admin Dashboard**:
   - Inspecting officers can confirm violations or flag false positives.
   - Admins have access to real-time compliance rate tracking, category violation breakdowns, 30-day timeline trends, most commonly missing fields, and **AI-resolved match logs** (`/dashboard/ai-matches`).

---

## Technology Stack

- **Framework**: FastAPI + Uvicorn
- **Database**: SQLite (`./data/app.db`) via SQLAlchemy ORM (swappable for PostgreSQL by updating `DATABASE_URL`)
- **OCR Engine**: PaddleOCR (mobile/lightweight CPU model, English + Hindi support)
- **Computer Vision**: OpenCV (`cv2`)
- **Authentication**: JWT (JSON Web Tokens) with direct `bcrypt` password hashing
- **Reporting**: ReportLab (PDF) & python-docx (DOCX)
- **Web Scraping**: Requests + BeautifulSoup4
- **Storage**: Local filesystem (`./uploads` and `./reports`)

## AI Usage & Gemini Configuration

This system utilizes the Google Gemini API for two distinct use cases to improve OCR reliability:
1. **Text field normalization (post-OCR)**: Resolves ambiguous or typo-laden extracted snippets by passing them to Gemini to deduce the matching regulatory field.
2. **Image inconsistency handling (vision)**: When PaddleOCR yields low-confidence or misses a mandatory field, the raw image panel is sent to Gemini's vision model (`gemini-3.8-flash`) to accurately read distorted, wrinkled, or glare-affected text.

To configure Gemini:
1. Obtain a free API key from [Google AI Studio](https://aistudio.google.com/).
2. Create a `.env` file in the root directory (you can copy `.env.example`).
3. Add `GEMINI_API_KEY=your_key_here` to the `.env` file.

*Note: If the key is missing or the API is unavailable, the system safely falls back to "OCR-only mode" and gracefully flags missing fields for manual officer review.*

---

## Prerequisites & Setup

### 1. Python Environment
Python **3.11**, **3.12**, or **3.13** (64-bit).

```bash
# Clone the repository
cd "SIH Prototype"

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Important Note on First-Time OCR Startup**:
> On the very first scan or OCR execution, PaddleOCR automatically downloads the official mobile model weights (~15 MB) to `~/.paddlex/` or `~/.paddleocr/`. This happens once, locally, and requires no API keys or accounts. Subsequent scans run locally from cache.

### 3. Initialize & Seed Database

Run the database seed script to auto-generate tables, create the mandatory Rule 6 rulebooks, and set up default accounts:

```bash
python seed.py
```

**Default Credentials Created:**
- **Admin**:
  - Email: `admin@legalmetrology.gov.in`
  - Password: `Admin@12345`
- **Field Officer**:
  - Email: `officer@legalmetrology.gov.in`
  - Password: `Officer@12345`

---

## Running the Server

Start the FastAPI application with Uvicorn:

```bash
python main.py
```
Or:
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The interactive API documentation (Swagger UI) is available at:
👉 **`http://127.0.0.1:8000/docs`**

---

## Running Automated Tests

A comprehensive integration test suite verifies authentication, blur rejection, physical label OCR, e-commerce scraping, officer override, PDF/DOCX generation, and admin dashboard statistics:

```bash
python tests/test_system.py
```

---

## Sample cURL Commands

### 1. Authentication

#### Register a New Officer
```bash
curl -X POST "http://127.0.0.1:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Inspector Rajesh Kumar",
    "email": "rajesh.kumar@legalmetrology.gov.in",
    "password": "Password@123",
    "role": "officer",
    "department": "North Zone Enforcement Unit"
  }'
```

#### Login (Officer)
```bash
curl -X POST "http://127.0.0.1:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=officer@legalmetrology.gov.in&password=Officer@12345"
```
*Save the returned `access_token` for subsequent requests:*
```bash
export TOKEN="<YOUR_ACCESS_TOKEN>"
```

#### Login (Admin)
```bash
curl -X POST "http://127.0.0.1:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin@legalmetrology.gov.in&password=Admin@12345"
```

---

### 2. Scanning & Compliance Checking

#### Upload a Packaged Product Label Photo (`POST /scans/upload`)
Synchronously runs blur check, deskew, CLAHE contrast enhancement, PaddleOCR, field extraction, font-size anomaly detection, and rulebook comparison:

```bash
curl -X POST "http://127.0.0.1:8000/scans/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/packaging_label.jpg" \
  -F "product_category=general_retail" \
  -F "location=Big Bazaar, Connaught Place, New Delhi"
```

#### Scan from E-Commerce Product Listing (`POST /scans/from-url`)
Scrapes listing metadata, specifications, and descriptions:

```bash
curl -X POST "http://127.0.0.1:8000/scans/from-url" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.example-store.in/products/atta-5kg",
    "product_category": "food",
    "location": "Online Marketplace Inspection"
  }'
```

---

### 3. Scan Management & Review

#### Get Scan Details (`GET /scans/{scan_id}`)
```bash
curl -X GET "http://127.0.0.1:8000/scans/1" \
  -H "Authorization: Bearer $TOKEN"
```

#### List Officer's Scan History with Filters (`GET /scans/history`)
```bash
curl -X GET "http://127.0.0.1:8000/scans/history?page=1&page_size=10&category=general_retail" \
  -H "Authorization: Bearer $TOKEN"
```

#### Officer Review Override (`PATCH /scans/{scan_id}/results/{result_id}/review`)
Confirm an infraction or mark a false alarm:

```bash
curl -X PATCH "http://127.0.0.1:8000/scans/1/results/2/review" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "officer_override": "confirmed_violation"
  }'
```

---

### 4. Report Generation

#### Download PDF Inspection Report (`GET /scans/{scan_id}/report`)
```bash
curl -X GET "http://127.0.0.1:8000/scans/1/report" \
  -H "Authorization: Bearer $TOKEN" \
  --output "Inspection_Report_Scan_1.pdf"
```

#### Download Word (DOCX) Inspection Report (`GET /scans/{scan_id}/report/docx`)
```bash
curl -X GET "http://127.0.0.1:8000/scans/1/report/docx" \
  -H "Authorization: Bearer $TOKEN" \
  --output "Inspection_Report_Scan_1.docx"
```

---

### 5. Admin Dashboard Statistics

#### Aggregate Statistics & Analytics (`GET /dashboard/stats`)
*(Requires Admin Token)*

```bash
curl -X GET "http://127.0.0.1:8000/dashboard/stats" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

**Response Example:**
```json
{
  "total_scans": 24,
  "total_violations": 6,
  "overall_violation_rate_pct": 25.0,
  "category_violation_rates": {
    "general_retail": 28.57,
    "food": 20.0,
    "cosmetics": 0.0,
    "electronics": 0.0
  },
  "status_breakdown": {
    "processing": 0,
    "completed": 24,
    "failed": 0
  },
  "compliance_breakdown": {
    "pass": 118,
    "needs_review": 14,
    "fail": 6
  },
  "last_30_days_trend": [
    {
      "date": "2026-09-17",
      "total_scans": 24,
      "violations": 6
    }
  ],
  "most_commonly_missing_fields": [
    {
      "field_name": "country_of_origin",
      "count": 4,
      "percentage": 16.67
    },
    {
      "field_name": "consumer_care",
      "count": 2,
      "percentage": 8.33
    }
  ]
}
```

---

## Directory Architecture

```
SIH Prototype/
├── app/
│   ├── config.py              # Application settings and pipeline thresholds
│   ├── database.py            # SQLite SQLAlchemy connection & session manager
│   ├── models/                # SQLAlchemy ORM models (User, Scan, ScanResult, Rulebook)
│   ├── schemas/               # Pydantic schemas (Requests, Responses, Enums)
│   ├── services/
│   │   ├── auth_service.py    # Direct bcrypt hashing & PyJWT verification
│   │   ├── image_service.py   # Blur detection (Laplacian), deskew, CLAHE
│   │   ├── ocr_service.py     # PaddleOCR local inference & bounding box mapper
│   │   ├── extraction_service.py # Rule 6 Regex & keyword extraction logic
│   │   ├── rule_checker.py    # Font size checking, confidence scoring, pass/fail rules
│   │   ├── scraper_service.py # Requests + BeautifulSoup listing parser
│   │   └── report_service.py  # ReportLab (PDF) & python-docx (DOCX) generators
│   └── routers/
│       ├── auth.py            # /auth/register, /auth/login
│       ├── scans.py           # /scans/upload, /scans/from-url, history, review, reports
│       └── dashboard.py       # /dashboard/stats (Admin only)
├── data/                      # Local SQLite file database (./data/app.db)
├── uploads/                   # Stored packaging photos
├── reports/                   # Generated inspection reports (.pdf & .docx)
├── tests/
│   └── test_system.py         # End-to-end automated integration test suite
├── seed.py                    # Database seeder (Rule 6 rulebooks & default users)
├── main.py                    # FastAPI entrypoint
├── api/
│   └── index.py               # Vercel Serverless Function entrypoint
├── vercel.json                # Vercel routing & build configuration
├── requirements.txt           # Vercel / Cloud production dependencies
└── requirements-local.txt     # Local development dependencies (+ PaddleOCR)
```

---

## Deploying to Vercel via GitHub

1. **Push to GitHub**:
   Ensure your code is pushed to your GitHub repository (e.g. `main` branch).

2. **Import Project in Vercel**:
   - Go to [vercel.com](https://vercel.com) and click **"Add New Project"**.
   - Select your GitHub repository.
   - Framework Preset: **Other** (or Leave default).
   - Root Directory: `./`

3. **Configure Environment Variables in Vercel**:
   - `GEMINI_API_KEY`: Your Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/).
   - *(Optional)* `JWT_SECRET_KEY`: A secure random string for JWT session encryption.
   - *(Optional)* `DATABASE_URL`: If using a persistent cloud database (e.g., Neon PostgreSQL / Supabase). Defaults to serverless `/tmp/app.db`.

4. **Deploy**:
   - Click **Deploy**. Vercel will install the serverless dependencies and provision the FastAPI backend.
   - Once deployed, visit your `https://your-project.vercel.app/` URL to open the Legal Metrology Compliance Scanner interface!
   - Default login accounts are automatically bootstrapped on startup:
     - **Admin**: `admin@legalmetrology.gov.in` / `Admin@12345`
     - **Officer**: `officer@legalmetrology.gov.in` / `Officer@12345`

