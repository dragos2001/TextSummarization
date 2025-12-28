# Text Summarization with mBART

This repository demonstrates a simple abstractive text summarization workflow using the Hugging Face `mBART` model. The accompanying notebook walks through loading the model and tokenizer, preparing Romanian input text, generating an abstractive summary, and evaluating it with ROUGE.

**Project Files**
- Notebook: [02_Code&Data/TextSummarization.ipynb](02_Code&Data/TextSummarization.ipynb)

**Model used**: `facebook/mbart-large-50-many-to-many-mmt` (mBART)

## Quick Setup

- Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install transformers torch evaluate rouge_score
```

Note: Use a GPU-enabled environment if available for faster generation.

## Usage

- Open and run the notebook at [02_Code&Data/TextSummarization.ipynb](02_Code&Data/TextSummarization.ipynb) for a step-by-step walkthrough.
- Minimal reproducible snippet (Python):

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

tokenizer.src_lang = "ro_RO"
tokenizer.tgt_lang = "ro_RO"

text = "România este un stat situat în sud-estul Europei Centrale..."
inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

summary_ids = model.generate(
	inputs["input_ids"],
	num_beams=2,
	max_length=50,
	do_sample=True,
	top_k=50,
	top_p=0.95,
	early_stopping=True,
	forced_bos_token_id=tokenizer.lang_code_to_id["ro_RO"],
)
summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
print(summary)
```

## Parameters used in the notebook
- `num_beams=2`
- `max_length=50`
- `do_sample=True`
- `top_k=50`
- `top_p=0.95`
- `early_stopping=True`

## Evaluation

The notebook demonstrates evaluating the generated summary using ROUGE via the `evaluate` package (rouge_score). Example code in the notebook shows computing ROUGE scores against a reference summary.

## Notes & Next Steps
- The notebook uses Romanian (`ro_RO`) as both source and target language. To summarize other languages, set `src_lang`/`tgt_lang` accordingly.
- Experiment with `num_beams`, `do_sample`, and length parameters to trade off fluency vs. faithfulness.
- Consider additional evaluation metrics and human evaluation for quality assessment.

## Credits
- Built with Hugging Face Transformers and the `evaluate` library.
