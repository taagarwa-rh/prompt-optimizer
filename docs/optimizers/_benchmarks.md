# Benchmarks

## Big Bench Hard - Hyperbaton (Adjective ordering)

**Split:** 50 train, 200 test

**Model:** Qwen/Qwen3-4B-Instruct-2507

**Optimizer Model:** openai/gpt-oss-20b

**Scoring Criteria:** Exact Match

**Prompt Selection Criteria:** Top Train Score

**Results:**

| Algorithm   | Train Score | Test Score | Time (min) | Prompts Generated |
| ----------- | ----------- | ---------- | ---------- | ----------------- |
| Baseline    | 0.58        | 0.645      | -          | -                 |
| APE         | 0.7         | 0.74       | 44.5       | **15**            |
| OPRO        | 0.74        | 0.78       | **21.2**   | 40                |
| PromptAgent | 0.62        | 0.7        | 171.4      | 39                |
| ProTeGi     | **0.8**     | **0.82**   | 436.2      | 73                |

### Prompt Comparison

**Sample Input:**

> Which sentence has the correct adjective order:
> Options:
> (A) midsize old grey Brazilian sweater
> (B) midsize grey Brazilian old sweater

**Sample Output:**

> (A)

**Baseline Prompt (0.645):**

> Answer the question with '(A)' or '(B)' only, and no other thoughts.

**ProTeGi Prompt (0.82):**

> Answer the question by keeping in mind the typical order of English adjectives: opinion, size, age, shape, color, origin, material, purpose.  
> Choose the option that follows this sequence and reply **only with** "(A)" or "(B)".

**OPRO Prompt (0.78):**

> Answer the question with '(A)' or '(B)' only, no other comments.
> 
> Which sentence has the correct adjective order:
> Options:
> (A) cold historic London treaty
> (B) historic cold London treaty
> 
> Output: (A)

**APE Prompt (0.74):**

> **Instruction**
> 
> For each question, you are given two candidate sentences that differ only in the order of the adjectives that modify the noun.  
> Select the sentence that follows the standard English adjective order:
> 
> ```
> opinion – size – age – shape – color – origin – material – (purpose)
> ```
> 
> Respond with the letter of the correct option – either **(A)** or **(B)**.  
> (Do not include any other text.)

**PromptAgent Prompt (0.7):**

> **Adjective‑Order Assistant (Strict Output Mode)**  
> 
> You are an English adjective‑order assistant.  
> The standard hierarchy for multiple adjectives in English is:  
> 
> **Opinion → Size → Age → Shape → Color → Origin → Material → Purpose**  
> 
> All sentences end with a noun that is **not** an adjective; that noun should stay in place.  
> If two or more adjectives belong to the same category, keep their relative order unchanged.  
> Purpose adjectives (e.g., *smoking, whittling, snorkeling*) must appear **after** all other categories and immediately before the noun.  
> 
> **Task**  
> When you receive a question that presents two adjective sequences, perform the following *internally* (do not write these steps out):  
> 1.  List every adjective and its category according to the hierarchy.  
> 2.  Re‑order each adjective sequence following the hierarchy, preserving intra‑category order.  
> 3.  Compare the original sequences to the reordered versions.  
>     - If exactly one sequence matches its reordered version, that sequence is correct.  
>     - If both match (i.e., both already obey the hierarchy), choose **(A)**.  
> 4.  Emit the final answer **and nothing else**.
> 
> **Final‑Answer Rule**  
> Your output must be **exactly** one of the two tokens:  
> ```
> (A)
> ```
> or  
> ```
> (B)
> ```
> Nothing else may appear—no spaces, no newlines, no surrounding text, and no extra characters.  
> If uncertainty remains about a word’s category, treat it as the closest plausible category; the hierarchy still applies.  
> 
> *Example*  
> Question: Which sentence has the correct adjective order?  
> Options:  
> (A) large archaic red Indonesian sock  
> (B) large red Indonesian archaic sock  
> 
> Answer: (A)  
> 
> Now perform the same process on the question supplied by the system.  
> 
> Your response: (A) or (B) only.  

