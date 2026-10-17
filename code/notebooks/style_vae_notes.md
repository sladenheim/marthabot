# Style-Conditioned Language Modeling — Notes

Companion reference for `StyleConditionedLanguageModeling.ipynb`.

**Goal:** train a small VAE on Shakespeare sentences so that sampling a latent
vector `z ~ N(0, I)` and decoding it produces Shakespeare-flavored text. This is
a teaching model — ~1.2M parameters, trains on CPU in ~25 minutes — not a system
meant to produce coherent prose.

---

## 1. The pipeline

```
data/shakespeare/*.nltktok          18,395 sentences, one per line
        │
        ▼  ShakespeareDataset        cell 1 — yields raw strings
        │
        ▼  Vocab.encode              cell 2 — words → integer ids
        │
        ▼  collate_fn                cell 2 — pad a batch into rectangular tensors
        │
        ▼  StyleVAE                  cell 5 — encode → z → decode
        │
        ▼  elbo_loss                 cell 6 — reconstruction + β·KL
        │
        ▼  train                     cell 7 — Adam, β annealing, per-epoch validation
        │
        ▼  generate                  cell 8 — sample z, decode to text
```

| cell | contents |
|------|----------|
| 1 | `ShakespeareDataset` — reads one sentence per line, returns raw `str` |
| 2 | `Vocab` + `collate_fn` — word-level tokenizer and batching |
| 3 | `DataLoader`s for train / validate / test |
| 4 | Hyperparameters and special-token ids |
| 5 | `StyleVAE` — the model |
| 6 | `elbo_loss`, `beta_schedule` |
| 7 | `evaluate`, `latent_report`, `train` |
| 8 | End-to-end run: train, sample, reconstruct, interpolate |

---

## 2. The data

Files are `.nltktok` — already word-tokenized by NLTK, meaning tokens are
space-separated and punctuation is split off as its own token:

```
I have a mind to strike thee ere thou speak'st .
```

Because it's pre-tokenized, "tokenizing" reduces to `str.split()`.

| split | sentences |
|-------|-----------|
| train | 18,395 |
| validate | 1,218 |
| test | 1,462 |

Total training corpus: **217,395 word tokens, 14,036 unique words**, mean
sentence length 11.8 words. This is a *very* small corpus, which drives most of
the design decisions below.

Each split also has a `.modern` counterpart (a modern-English paraphrase). We
use only `.original` — the Shakespeare side — since the goal is generating one
style, not translating between two.

---

## 3. `Vocab` — how words become numbers

A neural network can't consume strings, so every word is assigned an integer id.
`Vocab` builds that mapping from the **training split only** (using validation or
test data to build the vocabulary would leak information).

```python
self.itos = [PAD, BOS, EOS, UNK] + [w for w, _ in counts.most_common(max_vocab)]
self.stoi = {w: i for i, w in enumerate(self.itos)}
```

- `itos` — "**i**nteger **to** **s**tring", a list: `itos[5]` is the 5th word.
- `stoi` — "**s**tring **to** **i**nteger", the reverse dict.

### The four special tokens

Reserved at ids 0–3 so their values are known constants:

| token | id | purpose |
|-------|-----|---------|
| `<pad>` | 0 | filler so all sentences in a batch have equal length |
| `<bos>` | 1 | **b**eginning **o**f **s**equence — the seed token generation starts from |
| `<eos>` | 2 | **e**nd **o**f **s**equence — how the model signals "I'm done" |
| `<unk>` | 3 | **unk**nown — any word not in the vocabulary |

`<bos>` and `<eos>` matter for generation specifically. Generation starts with
nothing but `z`, so it needs *some* token to condition on — that's `<bos>`. And
it must decide when to stop, which it does by emitting `<eos>`. Without `<eos>`
the model would just run until it hit the token limit.

### Vocabulary size — and the `<unk>` trap

`MAX_VOCAB = None`, meaning **every word in the training split is kept**
(14,036 words + 4 specials = 14,040). It is worth understanding why, because
the obvious-looking alternative quietly ruins the output.

The first version capped the vocabulary at the 4,000 most frequent words, on the
standard reasoning that rare words appear too few times to learn anything from
and each one costs an embedding row plus an output-softmax column. That reasoning
is fine in general and wrong here:

| vocab | `<unk>` rate | `<unk>`'s frequency rank | embedding params |
|-------|--------------|--------------------------|------------------|
| 4,000 | **6.36%** | **#2 most frequent token** | 0.51M |
| 8,000 | 2.78% | #3 | 1.02M |
| 12,000 | 0.94% | #14 | 1.54M |
| **14,036 (all)** | **0.00%** | — | **1.80M** |

For comparison, the most frequent real tokens are `,` (8.33%), `.` (6.27%) and
`I` (2.38%). So at a 4,000-word cap, **`<unk>` was more frequent than the period
and nearly 3× more frequent than "I."**

The model then generated `<unk>` constantly — and it was *right* to. It had
faithfully learned the statistics of its training data, in which `<unk>` really
was the second most common event. The bug was in the data preparation, not the
model. This is a good general lesson: when generated output looks broken, check
whether the model is correctly reproducing something broken upstream.

Keeping every word costs 1.80M embedding parameters instead of 0.51M, taking the
whole model from 1.20M to 2.48M — still small enough to train on CPU.

**The one real caveat:** 7,102 words (51% of the vocabulary) appear *exactly
once*. Their embedding rows receive one or two gradient updates and stay
essentially random. They're nearly harmless, since a word seen once gets a tiny
probability and is almost never sampled, but they are dead parameters. A 12,000
cap (0.94% `<unk>`, rank #14) is a reasonable middle ground if you'd rather not
carry them.

Note that `<unk>` never disappears entirely: validation and test sentences
contain words absent from the *training* vocabulary, so it still shows up in the
`in :` lines of reconstructions. What changed is that the model no longer
*generates* it — see the `<unk>` ban in `generate` below.

### `encode` / `decode`

```python
vocab.encode("I love thee .")   # → [1, 6, 214, 40, 5, 2]
                                #    ^bos            ^eos
vocab.decode([1, 6, 214, 40, 5, 2])   # → "I love thee ."
```

`encode` wraps the sentence in `<bos> … <eos>` and truncates to `MAX_LENGTH`.
`decode` reverses it, stopping at the first `<eos>` and skipping padding.

---

## 4. `collate_fn` — assembling a batch

A `DataLoader` hands `collate_fn` a **list of raw strings** (whatever
`ShakespeareDataset.__getitem__` returned). Its job is to turn that list into
rectangular tensors, because a tensor must be a rectangle — but sentences have
different lengths.

```python
seqs = [vocab.encode(s) for s in batch]     # variable-length lists of ids
T = max(len(s) for s in seqs)                # longest in THIS batch
input_ids = torch.full((len(seqs), T), vocab.pad_id)
attention_mask = torch.zeros(len(seqs), T, dtype=torch.long)
for i, s in enumerate(seqs):
    input_ids[i, :len(s)] = torch.tensor(s)
    attention_mask[i, :len(s)] = 1
```

It returns two `(batch, T)` tensors:

- **`input_ids`** — the token ids, short sentences padded with `<pad>`.
- **`attention_mask`** — `1` at real tokens, `0` at padding.

### Why the mask is essential

Padding is meaningless filler. Without the mask, three things break:

1. The transformer would **attend to padding**, letting junk positions influence
   real tokens' representations.
2. The encoder's mean-pool would **average padding into the sentence vector**, so
   the same sentence would encode differently depending on what else was in its
   batch.
3. The loss would **train the model to predict `<pad>`**.

The mask is what makes all three a no-op. There's a test for property #2 in the
development history: append 6 pads to a batch and `mu` must not change.

### Why tokenize here rather than in the Dataset?

**Dynamic padding.** Each batch is padded to *its own* longest sentence, not to a
global `MAX_LENGTH=64`. Most sentences are ~12 tokens, so padding everything to
64 would waste ~5× the compute. Tokenizing per batch also lets the tokenizer run
on the whole batch at once, and keeps `ShakespeareDataset` simple — it just
returns strings.

---

## 5. `StyleVAE` — the model

A VAE has two halves:

- an **encoder** `q(z|x)` that compresses a sentence into a distribution over
  latent vectors, and
- a **decoder** `p(x|z)` that reconstructs a sentence from a latent vector.

Critically, the encoder outputs a *distribution*, not a point. That's what
separates a VAE from an ordinary autoencoder, and it's what makes the latent
space **samplable** — you can draw a random `z` and get valid output.

### Configuration (cell 4)

| knob | value | meaning |
|------|-------|---------|
| `D_MODEL` | 128 | width of embeddings and hidden states |
| `N_HEAD` | 4 | attention heads per layer |
| `N_LAYERS` | 2 | transformer layers in encoder, and in decoder |
| `D_FF` | 256 | feed-forward width inside each layer |
| `LATENT_DIM` | 32 | size of `z` |
| `DROPOUT` | 0.1 | standard dropout |
| `WORD_DROPOUT` | 0.25 | fraction of decoder input words blanked (see below) |

Total: **2.48M parameters.**

### The methods

#### `__init__`

Builds the embedding tables, the encoder and decoder stacks, and the projections
into and out of latent space. Two non-obvious details:

**Final LayerNorm.** The layers use `norm_first=True` (pre-LN), which requires a
`LayerNorm` *after* the stack. PyTorch only adds one if you pass `norm=`.
Omitting it lets activations grow unbounded — it doubled the initial loss.

**Small embedding init.** `nn.Embedding` defaults to `N(0, 1)`. Since `lm_head`
is *tied* to the embedding table (they share one weight matrix, halving
parameters), that default produces logits of magnitude ~128 — an absurdly
overconfident softmax before any training. `nn.init.normal_(std=0.02)` fixes it.

> **Sanity check worth remembering:** an untrained model's perplexity must equal
> the vocabulary size, because it should be guessing uniformly — so step 0 should
> print roughly **14,040**. Before these two fixes it printed `2.8e32`. This one
> check catches an entire class of initialization bug instantly, and it is the
> first thing to look at in any run.

#### `_embed(ids)`

Looks up token embeddings and adds learned position embeddings. Transformers have
no inherent notion of order — attention is permutation-invariant — so position
information must be added explicitly.

#### `encode(input_ids, attention_mask) → (mu, logvar)`

Runs the transformer encoder, then **mean-pools over real tokens** to get one
vector per sentence, then projects that to two vectors: `mu` and `logvar`.

Together they define a Gaussian `q(z|x) = N(mu, sigma²)` — the model's belief
about which latents could have produced this sentence.

**Why `logvar` instead of `sigma`?** Variance must be positive. A network can
output any real number, so predicting `log sigma²` and exponentiating guarantees
positivity for free, with no clamping and better numerical behavior.

**Why mean-pool?** We need a fixed-size vector from a variable-length sentence.
BERT uses a trained `[CLS]` token; we have no pretrained model, so averaging is
the robust default. The mask ensures padding is excluded.

#### `reparameterize(mu, logvar) → z`

$$z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

**Why this exists — the reparameterization trick.** We need to sample `z` from
`q(z|x)`, but *sampling is not differentiable*: you can't backpropagate through
a random draw, so the encoder would never receive a gradient.

The trick moves the randomness out of the path. Instead of "sample from
`N(mu, sigma²)`", we sample `epsilon` from a *fixed* `N(0, I)` and compute
`mu + sigma * epsilon`. The result is distributed identically, but now `mu` and
`sigma` are reached by ordinary arithmetic, so gradients flow through them. The
randomness sits in `epsilon`, which needs no gradient.

This single trick is what makes VAEs trainable.

#### `decode(z, input_ids, attention_mask) → logits`

Projects `z` into a single "memory" vector of shape `(B, 1, d_model)` that the
transformer decoder **cross-attends** to, and runs the decoder over the input
tokens with a **causal mask**.

**Why the causal mask?** The decoder generates left to right. Position `t` may
only see positions `< t`. Without the mask the model could see the answer it's
meant to predict and would learn nothing. (There's a test for this: change the
last token and earlier logits must not move.)

**Why does `decode` need `input_ids` at all?** During training the decoder is
**teacher-forced**: it's shown the *real* previous words and asked to predict
each next word. This is far faster and more stable than feeding it its own
(initially garbage) predictions. Free-running generation is what `generate` does.

#### `_word_dropout(ids)`

Randomly replaces 25% of the decoder's *input* words with `<unk>`, leaving the
**targets untouched** — the model must still predict the real word.

**Why:** this is the standard fix for posterior collapse (Bowman et al. 2016).
The decoder can often predict each word from the previous word alone, which means
it never needs `z`. Blanking a quarter of those previous words removes the
shortcut, forcing it to rely on `z` instead. `<bos>` and padding are excluded,
and it's disabled in eval mode.

#### `forward(input_ids, attention_mask) → dict`

The full training path:

```python
mu, logvar = self.encode(input_ids, attention_mask)
z = self.reparameterize(mu, logvar)
dec_in = self._word_dropout(input_ids[:, :-1])   # feed x[:-1]
logits = self.decode(z, dec_in, attention_mask[:, :-1])
targets = input_ids[:, 1:].clone()               # predict x[1:]
targets[attention_mask[:, 1:] == 0] = -100       # ignore pads
```

**The shift** is the language-modeling setup: feed positions `0..T-1`, predict
positions `1..T`. So at each step the model predicts the *next* word.

`-100` is PyTorch's `ignore_index` default — `cross_entropy` skips those
positions entirely, so padding contributes no loss and no gradient.

#### `generate(num_samples, max_new_tokens, z, temperature, top_k)`

Free-running generation. Starts from `<bos>`, and at each step: decode, take the
last position's logits, sample a token, append, repeat until every sequence has
emitted `<eos>`.

If `z` is not supplied it's drawn from the prior `N(0, I)` — **this is the actual
goal of the project**: sample the latent space, get Shakespeare.

**`<unk>` is banned during generation.** Its logit is set to `-inf` before
sampling. `<unk>` is a placeholder meaning "a word outside the vocabulary" — it
is not a word, and emitting it is never correct. This is a one-line guard that
should arguably always be on, independent of vocabulary size.

Two sampling controls:

- **`temperature`** divides the logits. `< 1` sharpens the distribution
  (safer, more repetitive), `> 1` flattens it (more varied, more errors).
- **`top_k`** keeps only the `k` most likely words and zeroes the rest before
  sampling. Without it you're drawing from all 14,040 words at every step, so
  there's always a small chance of nonsense. `top_k=40` visibly cleans up output.

---

## 6. The loss

### The ELBO

We want to maximize the likelihood of the data, `log p(x)`, but that requires
integrating over all possible `z` — intractable. Instead we maximize a lower
bound, the **Evidence Lower BOund**:

$$\log p_\theta(x) \;\ge\; \underbrace{\mathbb{E}_{q_\phi(z|x)}\big[\log p_\theta(x \mid z)\big]}_{\text{reconstruction}} \;-\; \underbrace{D_{KL}\big(q_\phi(z|x) \,\|\, p(z)\big)}_{\text{regularizer}}$$

We minimize its negative:

$$\mathcal{L} \;=\; \underbrace{-\mathbb{E}_{q}\big[\log p_\theta(x|z)\big]}_{\texttt{recon}} \;+\; \beta \cdot \underbrace{D_{KL}\big(q_\phi(z|x) \,\|\, \mathcal{N}(0,I)\big)}_{\texttt{kl}}$$

The two terms pull in opposite directions, and that tension *is* the VAE:

- **Reconstruction** wants `z` to carry as much information about `x` as
  possible, so the decoder can rebuild the sentence.
- **KL** wants `q(z|x)` to look like the prior `N(0, I)` — which, taken to its
  extreme, means `z` carries *no* information at all.

The KL term is what makes sampling work. It forces encoded sentences to fill the
same region that `N(0, I)` covers, so a randomly drawn `z` lands somewhere the
decoder understands. Without it you have a plain autoencoder whose latent space
has arbitrary gaps, and random draws decode to garbage.

### Reconstruction term

The decoder is autoregressive, so the sentence probability factorizes:

$$-\log p_\theta(x \mid z) \;=\; -\sum_{t=1}^{T} \log p_\theta\big(x_t \mid x_{<t},\, z\big)$$

which is exactly cross-entropy between the logits and the shifted targets:

```python
recon = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                        ignore_index=-100, reduction="sum") / batch_size
```

`reduction="sum"` then `/ batch_size` gives **nats per sentence**, putting it on
the same scale as the KL term so they can simply be added.

### KL term

For a diagonal Gaussian against a standard normal, the KL has a closed form — no
sampling needed:

$$D_{KL}\big(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, I)\big) \;=\; -\frac{1}{2} \sum_{j=1}^{d} \Big(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\Big)$$

```python
kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # (B, latent_dim)
```

Reading it: the KL is zero exactly when `mu = 0` and `sigma = 1` for every
dimension — i.e. when the posterior equals the prior and `z` says nothing about
`x`. Any deviation costs.

### Perplexity

$$\text{PPL} \;=\; \exp\!\left(\frac{\text{total NLL in nats}}{\text{number of tokens}}\right)$$

Interpretable as "on average, how many words is the model effectively choosing
between?" A uniform model over 14,040 words has perplexity 14,040. Lower is better.

---

## 7. Glossary

### nats

A unit of information, like a bit but base *e* instead of base 2. A **bit** is
the information in a fair coin flip (probability ½, `log₂`); a **nat** is the
information in an event of probability `1/e`.

$$1 \text{ nat} = \frac{1}{\ln 2} \approx 1.443 \text{ bits}$$

Everything here is in nats because `F.cross_entropy` and the Gaussian KL formula
both use natural log — natural log keeps the calculus clean (`d/dx e^x = e^x`).

Converting makes the numbers concrete:

| quantity | nats | bits | meaning |
|----------|------|------|---------|
| untrained per-token loss | `ln 14040` = 9.55 | 13.78 | uniform guess over 14,040 words |
| the `free_bits` floor | `0.2 x 32` = 6.40 | 9.23 | `2^9.23` ≈ **600 distinguishable latents** |

That second row is the single most useful number in the project. If the KL settles
on its floor, `z` can encode only ~600 distinct "meanings" — against 18,395
training sentences. That is plenty to capture mood and sentence shape, and nowhere
near enough to identify a *specific* sentence. It is the direct explanation for
reconstructions that match in register but not in content.

### Posterior collapse

The characteristic failure of text VAEs: the KL term drops to ~0, the decoder
learns to ignore `z` entirely, and the model degenerates into a plain language
model. Sampling different latents then produces the *same* generic sentence —
the whole point of the project, silently broken.

It happens because an autoregressive decoder is strong enough to model text on
its own, so the cheapest way to reduce the loss is to zero out the KL and let the
decoder do all the work.

Measured directly in this notebook: with `free_bits=0`, KL fell `6.59 → 0.036`
within 400 steps.

### β (beta) and annealing

**Annealing** means gradually changing a hyperparameter during training rather
than fixing it. Here it's the KL weight `β`:

$$\beta(s) \;=\; \beta_{\max} \cdot \min\!\left(1, \; \frac{s}{S_{\text{warmup}}}\right)$$

```python
def beta_schedule(step, warmup_steps, max_beta=1.0):
    return max_beta * min(1.0, step / max(1, warmup_steps))
```

`β` ramps linearly from 0 to 1 over the first 1,000 steps, then stays at 1.

**Why:** at `β = 1` from the start, the KL term is strongest exactly when the
decoder is worst at using `z` — so the fastest way down is to collapse the KL
immediately, before `z` has become useful for anything. Starting at `β = 0` makes
the model a plain autoencoder that freely learns to use `z`; by the time KL
pressure arrives, `z` is load-bearing and worth keeping.

Note `β = 1` is the *true* ELBO. Any other value optimizes a different objective
(this is the "β-VAE" family) — annealing is a training schedule, not a change to
the final goal.

### free bits

A floor on how much KL each latent dimension is charged for:

$$D_{KL}^{\text{free}} \;=\; \sum_{j=1}^{d} \max\big(\lambda, \; \overline{D_{KL,j}}\big)$$

```python
kl_dim = kl_dim.clamp(min=free_bits)   # one line, in elbo_loss
```

Each dimension gets a small information budget it isn't penalized for — hence
"free". Once a dimension's KL falls below `λ`, `clamp` pins it and **the gradient
through it becomes zero**, so the optimizer has no incentive to push it lower.
Above the floor it's penalized normally.

Total floor = `free_bits × LATENT_DIM` = `0.2 × 32` = **6.4 nats**.

It complements β-annealing: annealing protects `z` early, free bits holds the
line permanently once `β` reaches 1.

**The knob:** raise it → better reconstruction, but the aggregate posterior
spreads away from `N(0, I)` and prior samples get worse. Lower it → cleaner
samples, until collapse makes every sample identical.

### active units

The count of latent dimensions whose KL is meaningfully above zero. A dimension
with `KL ≈ 0` has `q(z_j|x)` equal to the prior: it encodes nothing and is dead
weight.

**Why track it:** total KL can look healthy while the model actually crams
everything into two or three dimensions and wastes the other 29. Only a
per-dimension breakdown reveals that.

> ⚠️ **The implementation in cell 7 is currently uninformative.** `threshold=0.01`
> sits *below* the `free_bits=0.2` floor, so every dimension clears it by
> construction and the metric always reports `32/32`. To make it meaningful, set
> the threshold above the floor (e.g. `free_bits * 1.5`) or compute the report
> with `free_bits=0`.

### Interpolation

Take two real sentences, encode both to get `mu₀` and `mu₁`, then decode a series
of points along the straight line between them:

$$z(t) = (1-t)\,\mu_0 + t\,\mu_1, \qquad t \in [0, 1]$$

**Why it matters.** It's the cleanest test of whether the latent space is
*structured* or merely *memorized*. If the model has genuinely learned a smooth
space, intermediate points decode to valid, gradually-changing sentences. If it
has just memorized training examples at scattered points, the space between them
is empty and midpoints decode to garbage.

It also verifies the property that makes sampling work at all: a randomly drawn
`z` will almost never coincide with an encoded sentence, so the regions *between*
encoded points must be meaningful. Interpolation checks exactly that.

---

## 8. Reading your results

No measured numbers are recorded here — earlier runs used a 4,000-word vocabulary
and are not comparable to the current configuration. This is what to look at
instead, and what good and bad look like.

### Step 0: is the model initialized correctly?

Perplexity at the very first step should be **≈ 14,040** (the vocabulary size),
because an untrained model guesses uniformly. Anything wildly larger means the
logits are mis-scaled — check the embedding init and the final LayerNorm. This
check is cheap and catches real bugs.

### During training: is the posterior collapsing?

Watch the `kl` column. The floor is `free_bits x LATENT_DIM` = `0.2 x 32` = **6.4
nats**.

| what you see | what it means |
|--------------|---------------|
| KL falls to ~0 | **Collapse.** `z` is ignored; every sample will look the same. Only possible if `free_bits=0`. |
| KL sits exactly at 6.4 | Free bits is holding the latent open, but the model is using the bare minimum it is forced to. Expect weak reconstructions. |
| KL settles **above** 6.4 | The model has decided `z` is genuinely worth using. This is the goal, and the sign that word dropout is doing its job. |

`recon` should fall steadily. If it plateaus while validation loss starts rising,
training has converged and further epochs will only overfit.

### At the end: what each output tells you

- **Samples from the prior** — the actual objective. They should be grammatical
  and Shakespearean in register, and *different from each other*. Identical
  samples across different `z` is the visible signature of collapse. They will not
  be semantically coherent; at this scale that is expected, not a bug.
- **Reconstructions** — how much information `z` really carries. Matching register
  but not content means `z` is near its floor (see the nats table in §7).
- **Interpolation** — whether the latent space is smooth. Intermediate points
  should decode to valid, gradually changing sentences. Abrupt jumps to garbage
  mean the model memorized scattered points rather than learning a space.
- **Per-dimension KL** — a flat profile pinned at the floor means the floor is the
  binding constraint, and `free_bits` is the lever to raise.

### Things to try next

1. **Raise `free_bits`** to 0.5 or 1.0 if the per-dimension KL is flat at the
   floor. Expect better reconstruction, possibly worse prior samples — that
   trade-off is the central tension of a VAE and is worth seeing directly.
2. **Fix the active-units threshold** (see the warning in §7) so the metric
   reports something other than `32/32`.
3. **Stronger word dropout** (0.5), or inject `z` at every decoder step rather
   than only through cross-attention.
4. ~~Vocab 8,000 to halve the `<unk>` rate.~~ **Done** — went to the full
   14,036-word vocabulary and banned `<unk>` at generation time.
5. **Cyclical KL annealing** — repeat the beta ramp several times instead of once.

## 9. References

- Kingma & Welling (2013), *Auto-Encoding Variational Bayes* — the VAE and the
  reparameterization trick.
- Bowman et al. (2016), *Generating Sentences from a Continuous Space* — text
  VAEs, posterior collapse, word dropout, KL annealing, interpolation.
- Kingma et al. (2016), *Improving Variational Inference with Inverse
  Autoregressive Flow* — introduces free bits.
- Higgins et al. (2017), *β-VAE* — weighting the KL term.
