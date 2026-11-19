# AI Text Summarizer with Gradio

A free, API-key-less text summarizer that runs an open-source LLM (**Qwen2.5-3B-Instruct**) on a Google Colab GPU and serves it through a **Gradio** web app. Long documents are handled with a **map-reduce** strategy, and you choose the output style, length and creativity.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Gradio](https://img.shields.io/badge/UI-Gradio-ff7c00)

## Features

- **Three output styles:** short summary, bullet points, and "Explain like I'm 12"
- **Three lengths:** Short, Medium, Long
- **Creativity slider** (temperature 0 to 1, default 0.3)
- **Input options:** paste text, or upload a PDF / TXT file
- **Long-document support** through chunking and map-reduce summarization
- **Grounding prompts** that tell the model to use only facts from the source text
- **Run statistics:** input words and tokens, chunks, LLM calls, output words, time taken
- **No API key, no paid service:** everything runs on your own (free Colab) GPU

## How It Works

1. Load Qwen2.5-3B-Instruct in float16 on the GPU.
2. Read text from the text box or from a PDF/TXT file (`pypdf`).
3. Clean the text (broken lines, extra spaces, hyphenated words).
4. Count tokens with the model's own tokenizer.
5. **Short input (up to 2,500 tokens):** one LLM call with the chosen style prompt.
6. **Long input:** split into sentence-aligned chunks of about 1,500 tokens (**map**: each chunk becomes at most 8 bullet notes), compress the notes again if they are still too long (up to 3 rounds), then write the final output from the notes (**reduce**).
7. Show the result and run statistics in the Gradio interface.

### Key parameters

| Parameter | Value | Purpose |
|---|---|---|
| `DIRECT_LIMIT` | 2,500 tokens | Single-call limit |
| `MAX_DOC_TOKENS` | 40,000 tokens | Longer text is truncated (user is notified) |
| Chunk size | 1,500 tokens | Size of each map-step chunk |
| Notes per chunk | up to 8 bullets / 250 tokens | Map-step output |
| Compression rounds | up to 3 | Applied if combined notes still exceed the limit |

## Tech Stack

- Python 3, PyTorch (CUDA)
- Hugging Face Transformers and Accelerate
- `Qwen/Qwen2.5-3B-Instruct` (fallback: `Qwen/Qwen2.5-1.5B-Instruct`)
- `pypdf` for PDF text extraction
- Gradio for the web interface

## Requirements

| Component | Used |
|---|---|
| GPU | NVIDIA Tesla T4 (16 GB), Google Colab |
| Model download | about 6 GB (first run only, 2 to 3 minutes) |
| Network | Internet for the first download; no API key |

## Getting Started

### Option 1: Google Colab (recommended)

1. Open the notebook `AI_Text_Summarizer_LLM_Gradio.ipynb` in Google Colab.
2. Set **Runtime > Change runtime type > T4 GPU**.
3. Run all cells.
4. Open the public `gradio.live` link printed at the end.

### Option 2: Local machine with an NVIDIA GPU

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install torch transformers accelerate pypdf gradio
jupyter notebook AI_Text_Summarizer_LLM_Gradio.ipynb
```

If GPU memory is not enough for the 3B model, switch the model name to `Qwen/Qwen2.5-1.5B-Instruct`.

## Usage

1. Paste text into the box, or upload a PDF/TXT file (the file takes priority).
2. Pick an output style and a length.
3. Adjust the creativity slider if you want.
4. Click **Summarize**.
5. Read the result and open the **Details** panel for statistics.

## Results

Test input: a 262-word article ("Sleep and Learning: Why a Good Night Matters"), Medium length, temperature 0.3, one LLM call per run.

| Style | Time | Output |
|---|---|---|
| Short summary | 10.3 s | 90 words, one paragraph |
| Bullet points | 7.3 s | 85 words, 7 bullets |
| Explain like I'm 12 | 15.0 s | 155 words, analogy style |

Run time mainly depends on how many tokens are generated. Length targets are soft guidance because they are controlled through prompts.

## Limitations

- Scanned (image-only) PDFs are not supported because there is no OCR.
- Grounding relies on prompting, so a small 3B model can still make occasional mistakes.
- Documents over 40,000 tokens are truncated.
- Map-reduce can lose fine detail, since the final step sees notes rather than the original text.
- The Gradio public link is temporary (about one week).
- The long-document path has not been benchmarked yet, and no automatic metrics (ROUGE, BERTScore) are computed.

## Future Scope

- OCR (for example Tesseract) for scanned PDFs
- Multilingual input and output
- Evaluation with ROUGE, BERTScore and human ratings
- 4-bit quantization or a larger model
- Token streaming and summary download
- Permanent hosting on Cloud Run or Hugging Face Spaces

## Project Structure

```
.
├── AI_Text_Summarizer_LLM_Gradio.ipynb   # main notebook (model, map-reduce, Gradio app)
├── AI_Text_Summarizer_Project_Report.pdf # full project report
├── AI_Text_Summarizer_IEEE_Paper.docx    # IEEE-format paper
└── README.md
```

## Author

**Sonu Kumar**
School of Computer Applications, Dayananda Sagar University

## License

Add a license of your choice (for example MIT) before publishing. Note that the Qwen2.5-3B-Instruct model has its own license terms on Hugging Face.

## Acknowledgements

- Qwen Team, Qwen2.5 models
- Hugging Face Transformers
- Gradio
- pypdf
