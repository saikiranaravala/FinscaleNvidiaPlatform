| Model | Valid JSON | Category accuracy | Category macro-F1 | Category weighted-F1 | Severity accuracy |
|---|---|---|---|---|---|
| Base | 100.0% (95% CI 100.0%-100.0%) | 65.0% (95% CI 45.0%-85.0%) | 0.634 (95% CI 0.316-0.713) | 0.642 | 60.0% (95% CI 35.0%-85.0%) |
| Fine-tuned | 100.0% (95% CI 100.0%-100.0%) | 75.0% (95% CI 55.0%-95.0%) | 0.624 (95% CI 0.310-0.714) | 0.727 | 95.0% (95% CI 85.0%-100.0%) |

### Base category confusion matrix (rows = gold, columns = predicted)

| gold \ pred | credit | cybersecurity | liquidity | market | operational | other | regulatory_legal | supply_chain |
|---|---|---|---|---|---|---|---|---|
| credit | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cybersecurity | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| liquidity | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| market | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 |
| operational | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| other | 0 | 0 | 1 | 0 | 0 | 1 | 2 | 1 |
| regulatory_legal | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 |
| supply_chain | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Fine-tuned category confusion matrix (rows = gold, columns = predicted)

| gold \ pred | credit | cybersecurity | liquidity | market | operational | other | regulatory_legal | supply_chain |
|---|---|---|---|---|---|---|---|---|
| credit | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cybersecurity | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| liquidity | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| market | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 |
| operational | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| other | 0 | 0 | 1 | 0 | 0 | 3 | 1 | 0 |
| regulatory_legal | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| supply_chain | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
