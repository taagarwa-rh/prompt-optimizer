# Examples

## Setup

Before running examples, you need to install the required dependencies:

```sh
uv sync --extra examples
```

## Available Examples

### [Hyperbaton](./hyperbaton.py)

Example showing how prompt optimization can be used to improve the performance of an AI system on the [BBH hyperbaton dataset](https://github.com/suzgunmirac/BIG-Bench-Hard).

The Hyperbaton task asks a language model to order english adjectives correctly:

```
Which sentence has the correct adjective order:
Options:
(A) midsize old grey Brazilian sweater
(B) midsize grey Brazilian old sweater

Answer: (A)
```

### [Benchmark](./benchmark)

Compare the performance of different optimizers on the same task.
