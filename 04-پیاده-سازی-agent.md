# پیاده‌سازی و اتصال Agent

---

## معماری کلی سیستم

<div dir="rtl">

بنچمارک endpoint نمی‌دهد. یعنی نمی‌توانی یک HTTP request بزنی و جواب بگیری.

</div>

```
Flask server  (محیط خرید)
       ↑ HTTP requests
WebShopEnv.py  ←  واسط بین agent و سرور
       ↑ Python function call
WebShop_test.py  ←  حلقه ارزیابی را اجرا می‌کند
       ↑ function call
model.py  ←  اینجا agent تو قرار می‌گیرد
```

---

## ReAct در عمل: جریان گام‌به‌گام

<div dir="rtl">

پیاده‌سازی ReAct در این محیط دقیقاً این‌طور است (جریان `test_direct`):

</div>

```
reset → observation₀
  ↓
model(observation₀) → "Thought: ...\nAction: search[...]"
  ↓
env.step(search_action) → observation₁
  ↓
model(observation₁, history) → "Thought: ...\nAction: click[ASIN]"
  ↓
env.step(click) → observation₂
  ↓
model(observation₂, history) → "Thought: ...\nAction: click[Buy Now]"
  ↓
env.step → reward
```

<div dir="rtl">

**نکته مهم:** مدل در هر گام فقط **یک کنش** برمی‌گرداند، نه یک توالی. تابع `search_find_buy_item` بازگشتی (recursive) است و هر بار یک کنش را execute می‌کند، بعد دوباره model را صدا می‌زند.

هر بار که model صدا زده می‌شود، history کامل مکالمه را دارد — پس مدل وضعیت قبلی را می‌بیند. این دقیقاً **ReAct** است: Reason → Act → Observe → Reason → ...

</div>

---

## اتصال Agent سفارشی: دو گام ساده

### گام ۱ — فایل جدید `Eval/my_agent.py`

```python
def my_generator(text, history=[], language='en', mode='direct'):
    # text = observation فعلی محیط
    # history = تاریخچه کامل مکالمه تا اینجا
    # mode = 'direct' | 'translate' | 'translate_en' | 'clp'

    # معماری هر چی دلت بخواد اینجا
    response = "Thought: ...\nAction: search[running shoes]"
    
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": response})
    return response, history
```

### گام ۲ — افزودن `elif` در `Eval/main.py:53`

```python
elif args.model == 'my_agent':
    from my_agent import my_generator
    test_model = WebShop_test(my_generator, saved_log_path)
```

### اجرا

```bash
python main.py --model my_agent --method direct --language en --test_n 10
```

---

## فرمت Observation دریافتی توسط Generator

<div dir="rtl">

متنی که به تابع generator می‌رسد، خروجی `get_clickable_actions` در `Eval/WebShop_test.py:55` است:

</div>

**صفحه init:**
```
Observation:
Instruction: i want to buy a red running shoe under $50
[Search]

Available Actions:
{'has_search_bar': True, 'clickables': "['Search']"}
```

**بعد از search:**
```
Observation:
[Back to Search]
Page 1 (Total results: 50)
[B07XYZ] Nike Running Shoe 
[B08ABC] Adidas Runner ...
[< Prev] [Next >]

Available Actions:
{'has_search_bar': False, 'clickables': "['Back to Search', 'B07XYZ', ...]"}
```

---

## مدل‌هایی که در پروژه واقعاً پیاده‌سازی شده‌اند

<div dir="rtl">

مدل‌های پیاده‌سازی‌شده در `model.py`:
- gpt-3.5-turbo
- gpt-4o
- deepseek_v2 (فقط اسمش در main.py هست — generator ندارد)
- deepseek-reasoner
- qwq
- llama3 (فقط یک print می‌زند — پیاده‌سازی نشده)

مدل‌های local در `Eval/os_model.py`:
- Qwen2-7B-Instruct — نیاز به GPU
- Mistral-7B-Instruct-v0.3 — نیاز به GPU

**نکته:** خود محیط تعداد معدود و مشخصی از مدل‌ها برای تست دارد — نشانه استاندارد بودن آزمایش.

</div>

---

## پرامپت‌ها

### پرامپت System (یک‌بار ابتدای مکالمه)

<div dir="rtl">

تابع اصلی `get_base_prompt` در `Eval/WebShop_prompt.py:80` است. مدل را در نقش یک خریدار وب قرار می‌دهد:

</div>

```
You are web shopping. I will give you instructions about what to do.
Every round I will give you an observation and a list of available actions...

search[<keyword>]
click[Prev]/click[Next]
click[Back to Search]
click[<item id>]
click[<attribute>]
click[Buy Now]

Your response should use the following format:
Thought: I think ...
Action: click[<something>]
```

<div dir="rtl">

**نکات مهم پرامپت:**
- کلمات `search` و `click` در پرامپت زبان‌محور هستند — برای فارسی/عربی/چینی با کلمه همان زبان جایگزین می‌شوند
- مثال‌های داخل پرامپت هم (earphone، black) بسته به زبان ترجمه‌شده‌اند
- هیچ chain-of-thought پیچیده‌ای ندارد — فقط یک `Thought: I think ...` ساده

</div>

### نمونه کلمات اکشن در زبان‌های مختلف (از `WebShop_prompt.py`)

```python
# انگلیسی:    search[keyword]        click[Buy Now]
# چینی:       搜索[keyword]           点击[现在购买]
# فرانسوی:    recherche[keyword]      cliquez sur[Acheter maintenant]
```

---

## روش CLP (Cross-Lingual Prompting)

<div dir="rtl">

**ایده اصلی:** مدل‌های زبانی در زبان‌های غیرانگلیسی ضعیف‌تر استدلال می‌کنند. CLP می‌گوید: اول به انگلیسی بفهم، بعد به زبان هدف عمل کن.

</div>

### مرحله ۱ — فهمیدن (`get_clp_1_prompt`)

```
Please act as an expert in multi-lingual understanding in [زبان].
You need fully understand every words, and explain its effection.
Don't response the Action, it is next step task.
Let's understand the observation in English step-by-step!
[observation]
```

### مرحله ۲ — عمل کردن (`get_clp_2_prompt`)

```
After understanding, you should act as an expert in reasoning in English.
In the end, you should response Action in [زبان] according previously understood observation.
Previous Action: [کنش قبلی]
[base_prompt]
```

### کد پیاده‌سازی در `WebShop_test.py:383`

```python
alignment, history = self.model(
    observation, history=[], language=language, mode='clp'
)  # مرحله ۱

action, history = self.model(
    get_clp_2_prompt(prev_action, language), history=history, language=language
)  # مرحله ۲
```

<div dir="rtl">

**نکته مهم:** در CLP، عامل history را نمی‌بیند و فقط اکشن قبلی به‌عنوان مرجع داده می‌شود (برخلاف BaseAgent که کل history را می‌بیند). این در کد با `history=[]` مشخص است.

**هزینه CLP:** چون در هر گام دو بار مدل صدا زده می‌شود، هزینه CLP تقریباً **دو برابر** متد direct است.

</div>

---

## تنظیم کلید API

<div dir="rtl">

تعیین کلید API جهت تست عامل در فایل `model.py`. امکان برنامه‌نویسی چارچوب در این فایل به سادگی امکان‌پذیر است.

برای اجرا:

</div>

```bash
python main.py --model gpt-4o-mini --method direct --language en --test_n 5
```

<div dir="rtl">

**خطای رایج:** اگر اسم مدل در لیست سخت‌گیرانه `main.py` نباشد، با خطای `UnboundLocalError` مواجه می‌شوید. برای مدل‌های جدید باید `elif` اضافه کنید.

</div>

---

## نتیجه اولین تست با GPT-3

<div dir="rtl">

عملکرد صفر. علت: مدل GPT-3 در ارائه‌دهنده خدمات استفاده‌شده موجود نبود.

**حداقل یادگیری:** مکانیزم اجرای مدل شناخته شد.

</div>

---

## باگ‌های شناخته‌شده در تست با GPT-4o

<div dir="rtl">

بررسی دقیق فایل log.json چند باگ ظریف در لایه‌ی تعامل مدل با بنچمارک آشکار کرد:

**۱. باگ عدم تغییر شماره صفحه در WebShop**

در صفحه ۲، کلیدهای موجود شامل `['Back to Search', '< Prev', 'Next >', ...]` هستند. مدل اکشن `click[Next >]` را انتخاب می‌کند تا به صفحه ۳ برود. اما در صفحه ۳، لیست محصولات کاملاً خالی است. وقتی مدل دوباره با کلمات جدید جستجو می‌کند و به صفحه ۲ می‌رسد، با زدن مجدد `click[Next >]`، از صفحه ۳ به بعد تا صفحه ۱۰، لیست محصولات دقیقاً روی همان آیتم‌های ثابت قفل می‌شود.

**۲. پدیده تعمیم افراطی کلمات کلیدی (Keyword Over-Generalization)**

مدل در گام پنجم، استراتژی خود را تغییر داده و کلمات توصیفی را به جستجو اضافه می‌کند: `search[JE-04B earbuds stereo sound comfortable under 70]`. در موتورهای جستجوی ضعیف مانند WebShop، اضافه کردن کلماتی مثل «comfortable» یا «under» دامنه‌ی جستجو را به شدت گسترش می‌دهد.

**۳. تضعیف توجه مدل در گام دهم (Context Fatigue)**

مدل gpt-4o در گام دهم، ساختار اولیه پرامپت را فراموش کرده و صرفاً کلمه تک‌کلمه‌ای `Error` را خروجی می‌دهد که منجر به crash نهایی پارسر پروژه شده است.

</div>

---

> **تصویر ۴:** نمایش متن تفکری که مدل تولید کرده است (Thought + Action)
