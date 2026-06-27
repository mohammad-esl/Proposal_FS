<div dir="rtl">

# تحلیل جامع پروژه ReAct

</div>

---

<div dir="rtl">

## ۱. موضوع پروژه

این پروژه پیاده‌سازی رسمی مقاله‌ی علمی زیر است:

</div>

> **ReAct: Synergizing Reasoning and Acting in Language Models**
> Published at ICLR 2023 — Yao et al.

<div dir="rtl">

ایده‌ی اصلی: مدل‌های زبانی بزرگ (LLM) معمولاً یا فقط **استدلال** می‌کنند (مثل Chain-of-Thought) یا فقط **عمل** می‌کنند (مثل ابزارهای ساده). ReAct این دو را با هم ترکیب می‌کند:

- **Re** → Reasoning (استدلال)
- **Act** → Acting (عمل‌کردن با ابزارها)

هر گام شامل سه مرحله است:
1. **Thought** — مدل استدلال می‌کند که باید چه کند
2. **Action** — مدل یک عمل انجام می‌دهد (جستجو، لوک‌آپ، یا پاسخ نهایی)
3. **Observation** — محیط نتیجه را برمی‌گرداند

</div>

---

<div dir="rtl">

## ۲. هر کد چه‌کار می‌کند

</div>

### `wikienv.py`

<div dir="rtl">

محیط ویکی‌پدیا برای تسک‌های QA و FEVER. این فایل یک محیط به سبک OpenAI Gym می‌سازد که سه اکشن را پشتیبانی می‌کند:

</div>

```python
search[entity]    # جستجوی ویکی‌پدیا برای یک موجودیت
lookup[keyword]   # پیدا کردن جمله‌ی بعدی حاوی کلیدواژه در صفحه‌ی فعلی
finish[answer]    # ارسال پاسخ نهایی
think[text]       # گام استدلال داخلی (برای ReAct prompting)
```

<div dir="rtl">

- `get_page_obs()`: اولین ۵ جمله از صفحه‌ی ویکی‌پدیا را برمی‌گرداند
- `construct_lookup_list()`: پاراگراف/جمله‌های مطابق کلیدواژه را پیدا می‌کند
- زمان‌سنجی API ویکی‌پدیا نیز در این فایل است

</div>

---

### `wrappers.py`

<div dir="rtl">

Wrapper های مختلف برای هر بنچمارک:

</div>

| کلاس | وظیفه |
|------|-------|
| `HistoryWrapper` | تبدیل observation به تاریخچه‌ی کامل trajectory برای in-context prompting |
| `HotPotQAWrapper` | بارگذاری داده‌های HotpotQA، محاسبه‌ی EM و F1 |
| `FeverWrapper` | بارگذاری داده‌های FEVER، محاسبه‌ی دقت طبقه‌بندی |
| `LoggingWrapper` | ذخیره‌ی trajectory ها در فایل JSON |

<div dir="rtl">

**نرمال‌سازی پاسخ** (برای مقایسه‌ی منصفانه):

</div>

```python
normalize_answer("The Boston Celtics") → "boston celtics"
# مراحل: lowercase → حذف مقالات → حذف نقطه‌گذاری → whitespace
```

---

### `prompts/`

<div dir="rtl">

فایل‌های JSON حاوی نمونه‌های few-shot برای هر تسک:

</div>

| فایل | تسک |
|------|-----|
| `prompts_naive.json` | HotpotQA — انواع مختلف prompting (ReAct / CoT / direct) |
| `fever.json` | FEVER — تشخیص SUPPORTS/REFUTES/NOT ENOUGH INFO |
| `alfworld.json` | AlfWorld — تسک‌های خانگی متنی |
| `alfworld_3prompts.json` | AlfWorld — ۳ نمونه برای هر نوع تسک |

<div dir="rtl">

ساختار نمونه‌های HotpotQA:

</div>

```
Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women...
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844–1846) was an American literary magazine...
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine...started in 1989.
Thought 3: Arthur's Magazine started in 1844, First for Women in 1989. So Arthur's Magazine started first.
Action 3: Finish[Arthur's Magazine]
```

---

### Notebooks

| فایل | تسک | مدل |
|------|-----|-----|
| `hotpotqa.ipynb` | پرسش‌وپاسخ چندگامه | `text-davinci-002` |
| `FEVER.ipynb` | تأیید ادعا | `text-davinci-002` |
| `alfworld.ipynb` | تسک‌های خانگی متنی | `text-davinci-002` |
| `WebShop.ipynb` | خرید آنلاین | `text-davinci-002` |

---

### `base_config.yaml`

<div dir="rtl">

فایل پیکربندی AlfWorld شامل:
- مسیرهای دیتاست (train / valid_seen / valid_unseen)
- تنظیمات محیط ALFRED (متنی یا بصری)
- روش آموزش: DAGGER یا DQN
- هایپرپارامترهای مدل و RL

</div>

---

<div dir="rtl">

## ۳. ارتباط کدها

</div>

```
wikienv.py  ──────────────────────────────────────────────────────────►  محیط پایه
     │
     ▼
wrappers.py  (HotPotQAWrapper / FeverWrapper / HistoryWrapper / LoggingWrapper)
     │                          │
     ▼                          ▼
  داده‌ها                    متریک‌ها
(data/*.json)           (EM, F1, reward)
     │
     ▼
  Notebooks  ◄──── prompts/*.json  (few-shot demonstrations)
(hotpotqa.ipynb / FEVER.ipynb / alfworld.ipynb / WebShop.ipynb)
     │
     ▼
  OpenAI API (text-davinci-002)
```

<div dir="rtl">

جریان اجرا به‌صورت ساده:
۱. Notebook داده می‌خواند
۲. سؤال + prompt few-shot → LLM
۳. LLM یک Thought/Action تولید می‌کند
۴. Action توسط `WikiEnv` اجرا می‌شود
۵. Observation به LLM برگردانده می‌شود
۶. این چرخه تا رسیدن به `Finish[...]` تکرار می‌شود

</div>

---

<div dir="rtl">

## ۴. ارزیابی چطور انجام می‌شود

</div>

<div dir="rtl">

**روش کلی:**
- برای HotpotQA و FEVER: نمونه‌برداری تصادفی ۵۰۰ نمونه از دیتاست dev (seed=233)
- برای AlfWorld: ارزیابی روی تمام ۱۳۴ بازی
- برای WebShop: ارزیابی کامل

**فرآیند ارزیابی یک نمونه:**

</div>

```python
env.reset(idx)          # لود کردن سؤال/تسک
for i in range(max_steps):
    action = llm(prompt + history)   # تولید action
    obs, reward, done, info = env.step(action)
    if done:
        break
metrics = env.get_metrics(answer)    # محاسبه EM / F1
```

---

<div dir="rtl">

## ۵. چطور می‌شود روش عامل خودم را تست کنم

برای تست عامل خودت، این مراحل را طی کن:

</div>

**گام ۱: نصب پیش‌نیازها**

```bash
pip install gymnasium wikipedia openai
# برای AlfWorld:
pip install alfworld
```

**گام ۲: تنظیم API Key**

```python
import openai
openai.api_key = "YOUR_API_KEY"
```

**گام ۳: ساخت محیط**

```python
import wikienv, wrappers

env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)
```

**گام ۴: نوشتن عامل خودت**

```python
def my_agent(question, env):
    obs, info = env.reset(idx)
    for step in range(7):
        # اینجا منطق عامل خودت را بنویس
        action = your_custom_policy(obs, question)
        obs, reward, done, info = env.step(action)
        if done:
            break
    return env.get_metrics(info['answer'])
```

**گام ۵: اجرا روی دیتاست**

```python
import random
random.seed(233)
idxs = random.sample(range(len(env.data)), 500)
results = [my_agent(env.data[i]['question'], env) for i in idxs]
print(f"EM: {sum(r['em'] for r in results)/len(results):.3f}")
```

<div dir="rtl">

می‌توانی prompt خودت را جایگزین `webthink_simple6` کنی یا از مدل دیگری غیر از `text-davinci-002` استفاده کنی.

</div>

---

<div dir="rtl">

## ۶. عامل مرجع چیست

</div>

<div dir="rtl">

عامل مرجع (baseline) چند نوع است که در `prompts_naive.json` تعریف شده‌اند:

</div>

| نام | توضیح |
|-----|-------|
| `webthink_simple6` | **ReAct کامل** — Thought + Action + Observation (مرجع اصلی) |
| `cotqa_simple6` | Chain-of-Thought فقط — بدون جستجو |
| `webqa_simple` | Direct QA — فقط پاسخ مستقیم، بدون استدلال |
| `webact_simple6` | Action-only — فقط عمل بدون Thought صریح |

<div dir="rtl">

مدل زبانی مرجع: **GPT-3 text-davinci-002** با temperature=0 و max_tokens=100 در هر گام

</div>

---

<div dir="rtl">

## ۷. محلی اجرا می‌شود یا ابری

</div>

<div dir="rtl">

**ترکیبی است:**

- **محیط (environment):** کاملاً **محلی** — WikiEnv، wrappers، و داده‌ها روی ماشین خودت اجرا می‌شوند
- **مدل زبانی (LLM):** **ابری** — تماس با OpenAI API از راه دور
- **داده‌ها:** **محلی** — فایل‌های JSON/JSONL روی دیسک

جهت اجرای کامل محلی، می‌توان مدل را با یک LLM محلی (مثل LLaMA، Mistral) جایگزین کرد.

</div>

---

<div dir="rtl">

## ۸. داده چیست و از کجا می‌آید

</div>

| بنچمارک | منبع | فایل محلی |
|---------|------|-----------|
| HotpotQA | [hotpotqa.github.io](https://hotpotqa.github.io) | `data/hotpot_dev_v1_simplified.json` |
| FEVER | [fever.ai](https://fever.ai) | `data/paper_dev.jsonl` |
| AlfWorld | [alfworld.github.io](https://alfworld.github.io) | از طریق پکیج `alfworld` |
| WebShop | [webshop.nlp.cs.princeton.edu](https://webshop-pnlp.github.io) | محیط آنلاین/شبیه‌ساز |

---

<div dir="rtl">

## ۹. نوع داده

</div>

**HotpotQA** — فرمت JSON:
```json
{
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "answer": "Arthur's Magazine",
  "type": "comparison"
}
```

**FEVER** — فرمت JSONL:
```json
{"claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.", "label": "SUPPORTS"}
```

<div dir="rtl">

- **HotpotQA:** متن آزاد — سؤال و پاسخ کوتاه
- **FEVER:** ادعای متنی با برچسب سه‌کلاسه (SUPPORTS / REFUTES / NOT ENOUGH INFO)
- **AlfWorld:** دستورات متنی و وضعیت محیط خانگی
- **WebShop:** توصیف محصول و دستور خرید به زبان طبیعی

</div>

---

<div dir="rtl">

## ۱۰. نسبت داده‌ها

</div>

**HotpotQA:**

| مجموعه | تعداد | استفاده در پروژه |
|--------|-------|-----------------|
| Train | ۹۰,۴۴۷ | نمونه‌های few-shot prompt |
| Dev | ۷,۴۰۵ | ارزیابی (۵۰۰ نمونه تصادفی) |
| Test | ۷,۴۰۵ | ارزیابی |

**FEVER:**

| مجموعه | تعداد | استفاده |
|--------|-------|---------|
| Dev | ۱۹,۹۹۸ | ارزیابی (۵۰۰ نمونه تصادفی) |

**AlfWorld:**

| مجموعه | تعداد | استفاده |
|--------|-------|---------|
| Train | ۳,۵۵۳ | ارجاع |
| Valid Seen | ۱۴۰ | ارزیابی |
| Valid Unseen | ۱۳۴ | **ارزیابی اصلی** (out-of-distribution) |

<div dir="rtl">

انواع تسک AlfWorld:

</div>

| نوع تسک | توضیح |
|---------|-------|
| `pick_and_place` | برداشتن شیء و گذاشتن در جای مشخص |
| `pick_clean_then_place` | تمیز کردن شیء سپس گذاشتن |
| `pick_heat_then_place` | گرم کردن شیء سپس گذاشتن |
| `pick_cool_then_place` | سرد کردن شیء سپس گذاشتن |
| `look_at_obj` | بررسی شیء زیر نور مشخص |
| `pick_two_obj` | برداشتن دو شیء هم‌نوع |

---

<div dir="rtl">

## ۱۱. معیار ارزیابی و محاسبه‌ی عملکرد عامل در یک تسک

</div>

### HotpotQA

**Exact Match (EM):**

```python
def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)
```

**F1 (Token-level):**

```python
# تقسیم پاسخ به توکن‌ها
# F1 = 2 * precision * recall / (precision + recall)
common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
num_same = sum(common.values())
precision = num_same / len(prediction_tokens)
recall = num_same / len(ground_truth_tokens)
f1 = (2 * precision * recall) / (precision + recall)
```

---

### FEVER

<div dir="rtl">

**Exact Match برچسب:**

</div>

```python
reward = 1 if normalize_answer(answer) == normalize_answer(label) else 0
# برچسب‌ها: "SUPPORTS" / "REFUTES" / "NOT ENOUGH INFO"
```

---

### AlfWorld

<div dir="rtl">

**نرخ موفقیت:**

</div>

```python
# یک بازی موفق است اگر در ≤ 50 گام به پایان برسد
success = 1 if reward > 0 and done else 0
success_rate = sum(successes) / total_games
```

---

### WebShop

<div dir="rtl">

**نرخ موفقیت خرید:**
- امتیاز نرمال‌شده بر اساس شباهت بین دستور خرید و محصول خریداری‌شده
- موفقیت کامل = ۱.۰، جزئی = مقدار بین ۰ و ۱

</div>

---

<div dir="rtl">

## نتایج عملکرد (از README)

</div>

| تسک | متریک | PaLM-540B | GPT-3 davinci-002 |
|-----|-------|-----------|-------------------|
| HotpotQA (500 dev) | EM | 29.4% | **30.4%** |
| FEVER (500 dev) | EM | **62.2%** | 54.0% |
| AlfWorld | Success Rate | 70.9% | **78.4%** |
| WebShop | Success Rate | **40.0%** | 35.8% |

---

<div dir="rtl">

## اطلاعات تکمیلی

### تنظیمات مدل زبانی

</div>

```python
llm_kwargs = {
    "model": "text-davinci-002",
    "temperature": 0,           # deterministic
    "max_tokens": 100,          # per step
    "stop": ["\nObservation"]   # توقف قبل از hallucinate کردن observation
}
```

<div dir="rtl">

### مقایسه روش‌های prompting

</div>

| روش | Thought | Action | نتیجه |
|-----|---------|--------|-------|
| ReAct | ✅ | ✅ | بهترین عملکرد کلی |
| CoT | ✅ | ❌ | خوب برای استدلال، بدون دسترسی به اطلاعات |
| Act-only | ❌ | ✅ | ممکن است در مسیریابی اشتباه کند |
| Direct QA | ❌ | ❌ | ضعیف‌ترین عملکرد |

<div dir="rtl">

### محدودیت‌های پروژه

- نیاز به **OpenAI API Key** دارد (هزینه‌دار)
- ممکن است API ویکی‌پدیا کُند باشد (retry logic تعبیه شده)
- AlfWorld نیاز به نصب پکیج جداگانه دارد
- WebShop نیاز به محیط شبیه‌ساز خارجی دارد

### مجوز

</div>

Apache License 2.0

---

<div dir="rtl">

*تاریخ تحلیل: ۱۴۰۵/۰۴/۰۵ — تحلیل توسط Claude Sonnet 4.6*

</div>
