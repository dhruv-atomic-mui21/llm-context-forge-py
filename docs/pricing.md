# Pricing Integrity & Versioned Registry

The cost estimation layer in `llm-context-forge` uses zero hardcoded prices in python code. All model pricing is loaded from a versioned YAML registry and verified against official provider documentation.

---

## Registry Schema

```yaml
gpt-4o:
  provider: openai
  input_per_1m: 2.50
  output_per_1m: 10.00
  cached_input_per_1m: 1.25
  context_window: 128000
  source_url: "https://openai.com/api/pricing"
  date_retrieved: "2026-07-18"
  confidence: official
  registry_version: "1.0.0"
```

## Runtime Staleness Warnings

When any model's `date_retrieved` is older than 30 days at runtime, a `PricingDataStaleWarning` is emitted:

```python
import warnings
# Emits: Pricing for gpt-4o was last verified 45 days ago...
```

## Local Enterprise Override

Override or extend pricing models using an environment variable:

```bash
export LLM_CONTEXT_FORGE_PRICING_FILE=/path/to/enterprise_pricing.yaml
```

## CLI Pricing Tools

```bash
# List all registered models with pricing and confidence
llm-context-forge pricing list

# Verify specific model rates
llm-context-forge pricing verify gpt-4o

# Update pricing from remote registry
llm-context-forge pricing update --remote
```
