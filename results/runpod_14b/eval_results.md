| Model | Valid JSON | Category accuracy | Category macro-F1 | Category weighted-F1 | Severity accuracy |
|---|---|---|---|---|---|
| Base | 100.0% (95% CI 100.0%-100.0%) | 73.1% (95% CI 61.5%-86.5%) | 0.526 (95% CI 0.302-0.614) | 0.767 | 67.3% (95% CI 53.8%-80.8%) |
| Fine-tuned | 100.0% (95% CI 100.0%-100.0%) | 73.1% (95% CI 61.5%-84.6%) | 0.478 (95% CI 0.323-0.595) | 0.775 | 90.4% (95% CI 80.8%-98.1%) |

### Base category confusion matrix (rows = gold, columns = predicted)

| gold \ pred | cybersecurity | liquidity | market | operational | other | regulatory_legal | supply_chain |
|---|---|---|---|---|---|---|---|
| cybersecurity | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| liquidity | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| market | 0 | 0 | 5 | 0 | 2 | 0 | 0 |
| operational | 0 | 0 | 5 | 17 | 0 | 0 | 0 |
| other | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| regulatory_legal | 0 | 1 | 1 | 0 | 0 | 14 | 0 |
| supply_chain | 0 | 0 | 0 | 0 | 0 | 1 | 0 |

### Fine-tuned category confusion matrix (rows = gold, columns = predicted)

| gold \ pred | cybersecurity | liquidity | market | operational | other | regulatory_legal | supply_chain |
|---|---|---|---|---|---|---|---|
| cybersecurity | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| liquidity | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| market | 0 | 0 | 5 | 0 | 2 | 0 | 0 |
| operational | 2 | 0 | 1 | 16 | 3 | 1 | 0 |
| other | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| regulatory_legal | 1 | 0 | 0 | 0 | 0 | 15 | 0 |
| supply_chain | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
