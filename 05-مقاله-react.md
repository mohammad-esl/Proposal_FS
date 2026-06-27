# مقاله ReAct: تحلیل دیتا و عامل پایه

---

## منشأ داده: نمونه‌برداری شده یا تولید شده؟

<div dir="rtl">

پاسخ کوتاه: **هر دو** — بسته به اینکه به کدام بخش داده اشاره داری.

</div>

### الف) داده ارزیابی (سؤال‌ها/ادعاها) → نمونه‌برداری شده

<div dir="rtl">

این‌ها تولید نمی‌شوند، بلکه از دیتاست‌های آماده و منتشرشده‌ی علمی نمونه‌برداری می‌شوند:

**روش نمونه‌برداری** (طبق مقاله، Appendix A.1 و بخش ۳):
- **HotpotQA و FEVER:** یک زیرمجموعه‌ی تصادفی ۵۰۰‌تایی از مجموعه‌ی validation/dev انتخاب می‌شود
- **ALFWorld:** نمونه‌برداری نیست؛ از تمام ۱۳۴ بازی unseen استفاده می‌شود
- **WebShop:** روی ۵۰۰ دستور تست ارزیابی می‌شود

</div>

```python
# در hotpotqa.ipynb سلول آخر:
idxs = list(range(7405))
random.Random(233).shuffle(idxs)   # بُر زدن قطعی با seed=233
...
for i in idxs[:500]:               # ۵۰۰ نمونه اول
    r, info = webthink(i, to_print=True)
```

<div dir="rtl">

«نمونه‌برداری» در عمل یعنی: شماره‌ی همه‌ی ۷٬۴۰۵ سؤال dev را با بذر ثابت 233 به‌هم می‌ریزند و ۵۰۰ تای اول را برمی‌دارند. بذر ثابت → نتیجه‌ی قابل‌بازتولید (هر بار همان ۵۰۰ سؤال).

</div>

### ب) داده few-shot prompt (نمونه‌های نمایشی) → تولیدشده (دستی، توسط انسان)

<div dir="rtl">

طبق بخش ۳.۲ مقاله:

> «For HotpotQA and Fever, we randomly select 6 and 3 cases from the training set and manually compose ReAct-format trajectories.»

یعنی:
- چند نمونه (۶ تا برای HotpotQA، ۳ تا برای FEVER) به‌صورت تصادفی از مجموعه‌ی train انتخاب می‌شوند
- سپس یک انسان (annotator) به‌صورت دستی مسیر Thought-Action-Observation را برای آن‌ها می‌نویسد
- این مسیرها به‌عنوان exemplar در prompt قرار می‌گیرند

**پاورقی مقاله:** «We find more examples do not improve performance» — بیشتر از این تعداد بهبودی نمی‌دهد.

</div>

```python
# در hotpotqa.ipynb:
webthink_examples = prompt_dict['webthink_simple6']   # ۶ نمونه دستی ReAct
webthink_prompt = instruction + webthink_examples
```

<div dir="rtl">

این مسیرهای دستی‌نوشته‌شده در `prompts/prompts_naive.json` ذخیره شده‌اند.

**مشاهدات هم تولید نمی‌شوند** بلکه زنده از API ویکی‌پدیا گرفته می‌شوند (`wikienv.py` خط ۱۰۰ به بعد).

</div>

---

## عامل پایه: چطور کار می‌کند؟

### فضای عمل (طبق بخش ۲ مقاله)

<div dir="rtl">

فضای عمل به Â = A ∪ L گسترش می‌یابد، که L فضای زبان است. یک عمل در فضای زبان (یک «thought») محیط را تغییر نمی‌دهد و observation برنمی‌گرداند؛ فقط context را به‌روز می‌کند.

</div>

### سه عمل واقعی روی ویکی‌پدیا (در `wikienv.py:124-160`)

```python
if action.startswith("search[")  ...   # ۵ جمله اول صفحه ویکی یا ۵ موجودیت مشابه
elif action.startswith("lookup[") ...   # جمله‌ی بعدی حاوی کلیدواژه (شبیه Ctrl+F)
elif action.startswith("finish[") ...   # پایان + ثبت پاسخ
elif action.startswith("think[")  ...   # فقط می‌گوید "Nice thought." — محیط تغییر نمی‌کند
```

<div dir="rtl">

**توجه:** `think[...]` همان «thought» مقاله است — observation معنادار برنمی‌گرداند (فقط "Nice thought.")، که دقیقاً منطبق بر تعریف نظری L در مقاله است.

</div>

### مدل زبانی استفاده‌شده

<div dir="rtl">

عامل یادگیری/آموزش نمی‌بیند (در حالت اصلی). مدل منجمد (frozen) است و فقط با few-shot prompting کار می‌کند. این کلید کار است: با ۱ تا ۶ مثال، از روش‌های imitation/RL که با ۱۰³ تا ۱۰⁵ نمونه آموزش دیده‌اند بهتر عمل می‌کند.

</div>

```python
# در hotpotqa.ipynb سلول ۱:
def llm(prompt, stop=["\n"]):
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt=prompt,
      temperature=0,        # قطعی (deterministic) → همان greedy decoding مقاله
      max_tokens=100,       # حداکثر ۱۰۰ توکن در هر گام
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=stop
    )
    return response["choices"][0]["text"]
```

### روش‌های prompting مقایسه‌شده (ablation از ReAct)

<div dir="rtl">

| روش | توضیح |
|-----|-------|
| ReAct | Thought + Action + Observation در هر گام |
| CoT | فقط chain-of-thought، بدون تعامل با محیط |
| Act-only | فقط Action، بدون Thought |
| CoT-SC | CoT با self-consistency |

</div>

---

## حلقه اصلی عامل: تابع `webthink`

```python
def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    question = env.reset(idx=idx)          # ۱. لود سؤال
    prompt += question + "\n"
    for i in range(1, 8):                  # ۲. حداکثر ۷ گام (7 steps for HotpotQA)
        # تولید Thought و Action با هم، توقف قبل از Observation
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:                            # ۳. اگر مدل بد فرمت داد، فقط Action را دوباره بگیر
            n_badcalls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(
                prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]
            ).strip()
        # ۴. اجرای Action در محیط
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str                 # ۵. افزودن به context برای گام بعد
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")   # ۶. اجباری پایان بده
    return r, info
```

---

## نکات کلیدی حلقه

<div dir="rtl">

**۱. حداکثر ۷ گام** (`range(1, 8)`) — دقیقاً همان «7 steps for HotpotQA» در مقاله (برای FEVER می‌شود ۵).

**۲. Stop sequence هوشمند:** مدل را وادار می‌کند فقط Thought و Action تولید کند و قبل از Observation متوقف شود — چون Observation باید از محیط واقعی بیاید، نه از توهم مدل. این همان نکته‌ای است که مقاله می‌گوید ReAct را «grounded» و ضد-hallucination می‌کند.

**۳. مدیریت خطای فرمت (`n_badcalls`):** اگر مدل خروجی بدفرمت داد (نتوانست Thought/Action را جدا کند)، فقط Action را دوباره صدا می‌زند. این بخش در مقاله نیست ولی جزو جزئیات مهندسی عامل پایه است.

**۴. Context تجمعی:** کل تاریخچه (`prompt += step_str`) در هر گام به ورودی مدل اضافه می‌شود — این همان cₜ = (o₁, a₁, ..., oₜ) در فرمول‌بندی مقاله است.

**۵. پایان اجباری:** اگر در ۷ گام پاسخ نداد، `finish[]` خالی فرستاده می‌شود (reward صفر).

</div>

---

## جمع‌بندی عامل پایه ReAct

<div dir="rtl">

عامل پایه = یک LLM منجمد (در کد: GPT-3 davinci-002، temp=0) + یک حلقه‌ی ساده‌ی بدون آموزش که با few-shot prompt مدل را وادار می‌کند در هر گام یک Thought و یک Action تولید کند، Action را در محیط ویکی‌پدیا اجرا می‌کند، Observation واقعی را برمی‌گرداند و به context اضافه می‌کند — تا رسیدن به `finish[]` یا سقف ۷ گام.

</div>
