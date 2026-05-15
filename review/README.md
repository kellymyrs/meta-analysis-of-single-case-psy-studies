# Gold Dataset Review Notes

Current review files were generated with:

```bash
python3 -m scripts.sced_review
python3 -m scripts.report_pdf_alignment_issues
python3 -m scripts.suggest_review_values
```

Use `disagreements_with_suggestions.csv` as the main review file. Clean rows
where legacy coding matches LLM B are auto-confirmed with
`decision_status=llm_b_confirmed`. Other rows become gold only after you check
the paper and fill:

- `reviewed_value_json`
- `decision_status`
- `reviewer`
- `decision_reason`
- `evidence_quote`
- `page`

Start with `high_priority_review_batch.csv` to identify the highest-value rows,
but make the final edits in `disagreements_with_suggestions.csv` so the build
script can read them. Do not accept rows marked `alignment_issue` until the PDF
or gold mapping is fixed.

After review:

```bash
python3 -m scripts.sced_review --build-gold \
  --review-csv review/disagreements_with_suggestions.csv \
  --gold-output data/sced_gold_reviewed_v1.jsonl

python3 -m scripts.evaluate_review_disagreements \
  --review-csv review/disagreements_with_suggestions.csv \
  --reference reviewed \
  --output review/disagreement_evaluation.json
```

Current counts:

- Silver rows: 168
- Rows needing review: 1460
- Auto-confirmed legacy plus full-PDF agreement rows: 606
- Legacy plus full-PDF extraction agreement rows: 652
- High-confidence LLM consensus against legacy: 36 total, 30 without alignment issues
- All sources different: 537
- One source only: 224
- Likely PDF alignment issues: 6 rows in `pdf_alignment_issues.csv`
- Review rows annotated with alignment issues: 133
- Rows still needing PDF/manual review: 721

Check `pdf_alignment_issues.csv` before reviewing field values. Rows affected by
duplicate or mismatched PDFs should be fixed at the file/gold alignment level
first; otherwise the review sheet will make the LLM output look wrong for the
wrong reason.
