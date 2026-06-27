# تحلیل جامع پروژه WebShop

<div dir="rtl">

این سند تحلیل کاملی از پروژه WebShop را پوشش می‌دهد که برای پایان‌نامه کارشناسی ارشد تهیه شده است.

---

## فهرست مطالب

۱. [معرفی پروژه](#معرفی-پروژه)  
۲. [توضیح هر فایل کد](#توضیح-هر-فایل-کد)  
۳. [عملکرد کلی سیستم](#عملکرد-کلی-سیستم)  
۴. [نحوه مشاهده محیط توسط agent](#نحوه-مشاهده-محیط-توسط-agent)  
۵. [نحوه تست agent خودتان در این پروژه](#نحوه-تست-agent-خودتان)  
۶. [داده‌ها: حجم و مکان](#داده‌ها-حجم-و-مکان)  
۷. [نسخه‌های مختلف پروژه](#نسخه‌های-مختلف-پروژه)  
۸. [فرایند ارزیابی](#فرایند-ارزیابی)  
۹. [نحوه تولید داده](#نحوه-تولید-داده)  
۱۰. [اندازه‌گیری عملکرد مدل و Ground Truth](#اندازه‌گیری-عملکرد-مدل-و-ground-truth)  
۱۱. [ارتباط بین کدها](#ارتباط-بین-کدها)  
۱۲. [مدل‌ها: API یا محلی](#مدل‌ها-api-یا-محلی)  
۱۳. [ادغام یک agent با منطق کد متفاوت](#ادغام-یک-agent-با-منطق-کد-متفاوت)  
۱۴. [بومی‌سازی داده](#بومی‌سازی-داده)  
۱۵. [نکات ارزشمند اضافی](#نکات-ارزشمند-اضافی)  

---

## معرفی پروژه

**WebShop** یک محیط شبیه‌سازی‌شده تجارت الکترونیک است که توسط گروه Princeton NLP ساخته شده و در مقاله‌ای با عنوان زیر معرفی شده است:

> "WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents" — Yao et al.

**هدف اصلی:** آموزش و ارزیابی agent هایی که از طریق رابط وب، محصول مناسب را بر اساس دستورالعمل زبان طبیعی پیدا، سفارشی، و خریداری می‌کنند.

**آمار کلیدی:**
- ۱.۱۸ میلیون محصول واقعی از آمازون
- ۱۲,۰۸۷ دستورالعمل جمع‌سپاری‌شده از کاربران انسانی
- دو حالت مشاهده: HTML خام و متن ساده
- معیار ارزیابی چندبعدی (نوع، صفات، گزینه‌ها، قیمت)

</div>

---

## توضیح هر فایل کد

<div dir="rtl">

### ساختار کلی پوشه‌ها

</div>

```
webshop/
├── web_agent_site/          # پکیج اصلی محیط (17MB)
├── baseline_models/         # مدل‌های پایه ML (32MB)
├── transfer/                # انتقال به سایت‌های واقعی
├── run_envs/                # اسکریپت‌های نمونه اجرا
├── search_engine/           # زیرساخت جستجوی Lucene
├── tests/                   # مجموعه تست‌ها
├── requirements.txt
├── setup.sh
└── README.md
```

---

### `web_agent_site/` — محیط اصلی

<div dir="rtl">

#### `web_agent_site/app.py`
سرور Flask که محیط خرید را به صورت وب‌سایت واقعی ارائه می‌دهد. برای حالت HTML که agent با Selenium روی مرورگر واقعی کار می‌کند استفاده می‌شود. مسیرهای URL را برای صفحه جستجو، نتایج، جزئیات محصول، و صفحه خرید تعریف می‌کند.

#### `web_agent_site/utils.py`
توابع کمکی شامل:
- تعریف ثابت‌ها (مسیر داده‌ها، اندازه صفحه نتایج)
- انتخاب تصادفی ایندکس محصولات
- لاگ‌گذاری

#### `web_agent_site/envs/web_agent_text_env.py` (641 خط)
**مهم‌ترین فایل برای اتصال agent به محیط.**
دو کلاس اصلی:
- `SimServer`: شبیه‌سازی سرور، مدیریت session، اختصاص هدف، templating HTML
- `SimBrowser`: شبیه‌سازی مرورگر، درخواست‌ها به سرور، پارس کردن HTML به متن
- کلاس `WebAgentTextEnv`: رابط OpenAI Gym برای حالت متنی
  - `observation_mode`: می‌تواند `'html'`, `'text'`, `'text_rich'`, `'url'` باشد
  - `step(action)`: یک action اجرا کرده، مشاهده جدید، پاداش، و وضعیت پایان را برمی‌گرداند
  - `reset()`: محیط را با یک هدف جدید ریست می‌کند

#### `web_agent_site/envs/web_agent_site_env.py` (217 خط)
حالت HTML با Selenium WebDriver. برای تست واقعی‌تر روی مرورگر Chrome/Firefox. کندتر از حالت متنی است.

#### `web_agent_site/engine/engine.py` (362 خط)
موتور اصلی بازی:
- بارگذاری محصولات از فایل JSON
- پیاده‌سازی جستجو (Lucene یا BM25)
- تولید HTML صفحات مختلف (جستجو، نتایج، جزئیات، خرید)
- پارس کردن action های کلیک
- مدیریت گزینه‌های انتخاب‌شده توسط agent

#### `web_agent_site/engine/goal.py` (269 خط)
**فایل پاداش و Ground Truth:**
- ساختار داده هدف
- محاسبه پاداش چندبعدی:
  - `r_type`: تطابق دسته‌بندی و عنوان محصول
  - `r_att`: نسبت صفات هدف موجود در محصول خریداری‌شده
  - `r_option`: نسبت گزینه‌های سفارشی‌سازی درست انتخاب‌شده
  - `r_price`: آیا قیمت زیر سقف مجاز است؟
- تابع `get_reward()`: پاداش نهایی = `(r_att×w + r_opt×w + r_price×w) × r_type`

#### `web_agent_site/engine/normalize.py` (103 خط)
نرمال‌سازی رنگ‌ها و اندازه‌ها. مثلاً "small", "S", "sm" همه به یک مقدار استاندارد تبدیل می‌شوند. برای مقایسه صحیح گزینه‌های agent با هدف ضروری است.

#### `web_agent_site/models/models.py`
سه کلاس پایه policy:
- `BasePolicy`: رابط انتزاعی
- `HumanPolicy`: ورودی دستی از کنسول (برای تست انسانی)
- `RandomPolicy`: انتخاب تصادفی از بین action های موجود

</div>

---

### `baseline_models/` — مدل‌های پایه

<div dir="rtl">

#### `baseline_models/agent.py` (162 خط)
کلاس `Agent` که مدل BERT و BART را ترکیب می‌کند:
- `bart_model`: تولید query جستجو از دستورالعمل
- `bert_model`: انتخاب action از بین گزینه‌های موجود
- `encode_action()`: رمزگذاری state + action برای BERT
- `action_sampler()`: نمونه‌گیری action بر اساس logit ها

#### `baseline_models/env.py` (238 خط)
Wrapper روی `WebAgentTextEnv` برای مدل‌های پایه:
- اضافه کردن تاریخچه state (memory mode)
- فیلتر کردن action های معتبر
- مدیریت هدف‌ها برای آموزش (train/eval/test split)
- پاداش‌دهی step-based

#### `baseline_models/test.py` (140 خط)
**اسکریپت ارزیابی اصلی:**
- بارگذاری مدل‌های آموزش‌دیده (BERT + BART)
- اجرای ۵۰۰ episode روی مجموعه test
- گزارش: میانگین reward، نرخ موفقیت، harsh reward
- پشتیبانی از حالت‌های: greedy، softmax، rule-based

#### `baseline_models/train_choice_il.py` (629 خط)
آموزش مدل BERT به روش Imitation Learning:
- خواندن داده trajectory های انسانی
- تبدیل به batch های PyTorch
- Cross-entropy loss روی action های انتخابی انسان
- ذخیره checkpoint ها

#### `baseline_models/train_search_il.py` (139 خط)
آموزش مدل BART برای تولید query جستجو:
- Input: دستورالعمل زبان طبیعی
- Output: query جستجو
- Seq2seq با BART-large

#### `baseline_models/train_rl.py` (249 خط)
آموزش Reinforcement Learning بعد از IL:
- ۴ محیط موازی
- Policy gradient + TD loss + Entropy regularization
- لاگ‌گذاری با WandB

#### `baseline_models/models/bert.py`
`BertModelForWebshop`: معماری اصلی مدل انتخاب action:
- BERT encoder برای state و action
- BiAttention: attention دوطرفه بین state و action
- Token aggregation: میانگین گیری روی توکن‌های action
- Optional: ادغام feature های تصویری ResNet (512→768 dim)

#### `baseline_models/models/rnn.py`
`RCDQN`: مدل جایگزین بر پایه GRU:
- لایه‌های GRU برای encoding توالی
- Attention mechanism
- کمتر از BERT استفاده می‌شود

#### `baseline_models/logger.py` (542 خط)
سیستم لاگ‌گذاری جامع:
- خروجی JSON، CSV
- یکپارچه‌سازی با WandB و TensorBoard
- ذخیره metrics دوره‌ای

</div>

---

### سایر فایل‌ها

<div dir="rtl">

#### `run_envs/run_web_agent_text_env.py`
مثال تعاملی برای نشان دادن نحوه اتصال یک policy ساده به محیط متنی. برای درک API مناسب است.

#### `run_envs/run_web_agent_site_env.py`
مثال مشابه برای حالت HTML با Selenium.

#### `transfer/app.py`
دمو Gradio برای تست agent روی آمازون/eBay واقعی.

#### `transfer/predict_help.py`
Web scraping از آمازون و eBay برای مقایسه رفتار agent در محیط واقعی.

#### `search_engine/lucene_searcher.py`
Wrapper روی pyserini/Lucene برای جستجوی BM25 روی محصولات. ایندکس‌های مختلف را پشتیبانی می‌کند.

#### `search_engine/convert_product_file_format.py`
تبدیل فرمت فایل محصولات به فرمت قابل ایندکس توسط Lucene.

#### `setup.sh`
نصب خودکار وابستگی‌ها، دانلود داده‌ها، ساخت ایندکس‌های Lucene.

</div>

---

## عملکرد کلی سیستم

<div dir="rtl">

### معماری کلی

سیستم WebShop یک حلقه تعامل agent-محیط کلاسیک OpenAI Gym است:

</div>

```
[دستورالعمل زبان طبیعی]
         ↓
    Agent / Policy
         ↓
   action (search/click)
         ↓
   WebAgentTextEnv
         ↓
   SimBrowser → SimServer
         ↓
   Engine (جستجو، رندر صفحه)
         ↓
   observation (متن/HTML)
   reward (0.0 - 1.0)
   done (True/False)
         ↓
    Agent / Policy
         ↓ (تکرار تا done=True)
```

<div dir="rtl">

### جریان کامل یک episode

۱. **ریست محیط**: `env.reset()` — یک هدف (دستورالعمل + محصول هدف) انتخاب می‌شود
۲. **تولید query**: BART مدل دستورالعمل را به query جستجو تبدیل می‌کند  
۳. **جستجو**: `search[query]` — Lucene نتایج را برمی‌گرداند  
۴. **مرور نتایج**: agent روی محصولات کلیک می‌کند  
۵. **بررسی محصول**: جزئیات، توضیحات، و گزینه‌ها بررسی می‌شوند  
۶. **انتخاب گزینه‌ها**: رنگ، سایز، و سایر ویژگی‌ها انتخاب می‌شوند  
۷. **خرید**: `click[Buy Now]` — پاداش محاسبه و episode تمام می‌شود  

### فضای action
- `search[keywords]` — جستجوی متنی
- `click[product_name]` — کلیک روی محصول در نتایج
- `click[Description]`, `click[Features]`, `click[Reviews]` — مرور تب‌ها
- `click[option_value]` — انتخاب رنگ/سایز/...
- `click[Buy Now]` — خرید نهایی
- `click[< Prev]`, `click[Next >]` — صفحه‌بندی
- `click[Back to Search]` — بازگشت

### فضای مشاهده (observation modes)
- `'text'`: متن تمیز صفحه
- `'text_rich'`: متن با علامت‌گذاری دکمه‌ها با `[button]...[button_]`
- `'html'`: HTML خام
- `'url'`: فقط URL جاری

</div>

---

## نحوه مشاهده محیط توسط agent

<div dir="rtl">

agent در هر step یک **observation** دریافت می‌کند. این observation می‌تواند در چهار فرمت مختلف باشد که هنگام ساخت محیط انتخاب می‌شود. علاوه بر observation متنی، یک کانال جداگانه برای تصویر نیز وجود دارد.

### حالت‌های observation متنی

</div>

```python
env = gym.make('WebAgentTextEnv-v0', observation_mode='text_rich')
#  مقادیر مجاز: 'html' | 'text' | 'text_rich' | 'url'
```

---

#### حالت `'html'` — HTML خام

<div dir="rtl">

agent کل سورس HTML صفحه را می‌گیرد. تگ‌ها، کلاس‌ها، و ساختار DOM کامل در اختیار است.

</div>

```html
<!-- نمونه خروجی در صفحه جزئیات محصول -->
<html>
  <body>
    <div id="instruction-text"><h4>Buy a waterproof men's running shoe, size 9, black, under $80</h4></div>
    <div id="product-image"><img src="https://..." id="product-image"/></div>
    <h3>Nike Air Zoom Waterproof Running Shoe</h3>
    <p>Price: $74.99</p>
    <label><input type="radio" name="size" value="9"/> 9</label>
    <label><input type="radio" name="color" value="black"/> Black</label>
    <button class="btn">Buy Now</button>
    <button class="btn">Description</button>
    ...
  </body>
</html>
```

<div dir="rtl">

**مزیت:** بیشترین اطلاعات ممکن  
**معایب:** طولانی، پر از نویز، پردازش سخت‌تر برای LLM

</div>

---

#### حالت `'text'` — متن ساده

<div dir="rtl">

HTML به متن ساده تبدیل می‌شود. تمام تگ‌ها حذف شده و بخش‌های مختلف با `[SEP]` از هم جدا می‌شوند.

</div>

```
Buy a waterproof men's running shoe, size 9, black, under $80 [SEP] 
Nike Air Zoom Waterproof Running Shoe [SEP] 
Price: $74.99 [SEP] 9 [SEP] Black [SEP] Buy Now [SEP] Description [SEP] Features [SEP] Reviews
```

<div dir="rtl">

**مزیت:** کوتاه، تمیز، مناسب LLM با context window محدود  
**معایب:** تمایز بین دکمه و متن عادی از بین می‌رود

</div>

---

#### حالت `'text_rich'` — متن غنی‌شده (پیشنهادی)

<div dir="rtl">

مثل `text` اما با علامت‌گذاری خاص برای عناصر تعاملی. این حالت **برای اکثر agent های ML و LLM بهترین انتخاب است.**

- دکمه‌های کلیک‌نشده: `[button] Buy Now [button_]`
- دکمه‌های کلیک‌شده: `[clicked button] Black [clicked button_]`
- لینک محصولات کلیک‌شده نیز با `[clicked button]` علامت‌گذاری می‌شوند
- در ابتدای observation اعلام می‌شود که چه چیزی کلیک شده: `You have clicked Black.`

</div>

```
You have clicked black.
Buy a waterproof men's running shoe, size 9, black, under $80
Nike Air Zoom Waterproof Running Shoe
Price: $74.99
  [clicked button] black [clicked button_]
  [button] white [button_]
  [button] 8 [button_]
  [clicked button] 9 [clicked button_]
[button] Buy Now [button_]
[button] Description [button_]
[button] < Prev [button_]
```

<div dir="rtl">

**مزیت:** agent می‌داند کدام گزینه‌ها را قبلاً انتخاب کرده  
**معایب:** کمی طولانی‌تر از `text` ساده

</div>

---

#### حالت `'url'` — فقط URL

<div dir="rtl">

فقط آدرس URL صفحه جاری برگردانده می‌شود. URL اطلاعات زیادی درباره وضعیت دارد:

</div>

```
http://127.0.0.1:3000/item_page/abc123/B07XYZ/running+shoes/1/{"color": "black"}
```

<div dir="rtl">

**کاربرد:** معمولاً به تنهایی کافی نیست؛ بیشتر برای debug یا ترکیب با سایر حالت‌ها

</div>

---

### تصویر محصول (کانال جداگانه)

<div dir="rtl">

تصویر **بخشی از observation متنی نیست** بلکه از طریق متد جداگانه `env.get_image()` در اختیار agent قرار می‌گیرد.

**فرمت:** یک بردار `torch.Tensor` به شکل `(512,)` که feature های استخراج‌شده از ResNet است (نه pixel های خام)

**منبع:** از فایل‌های پیش‌محاسبه‌شده `data/feat_conv.pt` و `data/feat_ids.pt` لود می‌شود

**نحوه فعال‌سازی:**

</div>

```python
# فعال‌سازی هنگام ساخت محیط
env = gym.make('WebAgentTextEnv-v0', get_image=True)

# دریافت تصویر در هر step
image_feat = env.get_image()  # torch.Tensor shape: (512,)
# اگر محصول تصویر نداشته باشد: torch.zeros(512)
```

<div dir="rtl">

**نحوه استفاده در مدل BERT:**  
مدل پایه BERT می‌تواند این بردار را از طریق یک لایه projection به فضای 768 بعدی تبدیل کند و به عنوان یک توکن اضافی به ابتدای sequence اضافه کند (`image=True` در config):

</div>

```python
# در models/bert.py
config = BertConfigForWebshop(image=True)   # فعال‌سازی تصویر
model = BertModelForWebshop(config)
# مدل خودکار image_feat را در forward pass ادغام می‌کند
```

---

### دستورالعمل هدف (Instruction Text)

<div dir="rtl">

**این بخش نه بخشی از observation است، نه جداگانه** — بلکه **درون** همه حالت‌های observation (متن یا HTML) به صورت یک عنصر با `id="instruction-text"` تعبیه شده است.

agent می‌تواند آن را مستقیماً از observation استخراج کند یا از طریق:

</div>

```python
instruction = env.get_instruction_text()
# مثال: "Buy a waterproof men's running shoe, size 9, black, under $80"
```

---

### state کامل vs. observation

<div dir="rtl">

تفاوت مهم: `state` کامل‌تر از `observation` است.

</div>

```python
# state شامل همه اطلاعات است (برای debug)
state = env.state
# {
#   'url': 'http://127.0.0.1:3000/item_page/...',
#   'html': '<html>...</html>',        # همیشه HTML کامل
#   'instruction_text': 'Buy a ...'
# }

# observation فقط زیرمجموعه‌ای است که به agent داده می‌شود
obs = env.observation   # طبق observation_mode تنظیم‌شده
```

---

### تاریخچه (Memory Mode)

<div dir="rtl">

به صورت پیش‌فرض agent فقط observation جاری را می‌بیند. اما می‌توان observation ها و action های قبلی را هم به آن داد:

</div>

```python
env = gym.make(
    'WebAgentTextEnv-v0',
    num_prev_obs=2,      # اضافه کردن 2 observation قبلی
    num_prev_actions=2,  # اضافه کردن 2 action قبلی
)
# خروجی step(): obs_t [SEP] action_{t-1} [SEP] obs_{t-1} [SEP] action_{t-2} [SEP] obs_{t-2}
```

---

### خلاصه مقایسه حالت‌ها

</div>

| حالت | طول | اطلاعات دکمه | اطلاعات کلیک‌شده | مناسب برای |
|------|-----|--------------|-----------------|-----------|
| `html` | خیلی زیاد | بله (تگ) | بله (تگ) | HTML parser، LLM با context بزرگ |
| `text` | کوتاه | خیر | خیر | LLM با context محدود |
| `text_rich` | متوسط | بله `[button]` | بله `[clicked button]` | **اکثر موارد — پیشنهادی** |
| `url` | بسیار کوتاه | خیر | خیر | debug، ترکیبی |
| تصویر | ثابت (512) | — | — | مدل‌های multimodal |

---

## نحوه تست agent خودتان

<div dir="rtl">

برای اتصال agent سفارشی خودتان به این محیط، باید یک Policy بسازید که با API استاندارد Gym کار کند.

### روش ۱: مستقیم با WebAgentTextEnv (ساده‌ترین)

</div>

```python
# my_agent_test.py
import gym
import web_agent_site

# ساخت محیط
env = gym.make(
    'WebAgentTextEnv-v0',
    observation_mode='text',   # یا 'text_rich' یا 'html'
    num_products=1000,         # تعداد محصولات
    get_image=False            # بدون تصویر
)

def my_agent(observation, available_actions):
    """
    observation: متن صفحه فعلی
    available_actions: {'has_search_bar': bool, 'clickables': [...]}
    returns: action string مثل 'search[laptop]' یا 'click[Buy Now]'
    """
    # منطق agent شما اینجا
    # مثلاً یک LLM که observation را می‌گیرد و action برمی‌گرداند
    if available_actions['has_search_bar']:
        return 'search[my query]'
    clickables = available_actions['clickables']
    return f'click[{clickables[0]}]'

# حلقه ارزیابی
total_reward = 0
n_episodes = 500  # 500 goal برای test set

for i in range(n_episodes):
    obs = env.reset()
    done = False
    while not done:
        available_actions = env.get_available_actions()
        action = my_agent(obs, available_actions)
        obs, reward, done, info = env.step(action)
    total_reward += reward

print(f"Average Reward: {total_reward / n_episodes:.4f}")
```

<div dir="rtl">

### روش ۲: ادغام با ساختار baseline_models (پیشرفته‌تر)

برای ادغام با pipeline آموزشی و ارزیابی موجود، از `WebEnv` در `baseline_models/env.py` استفاده کنید:

</div>

```python
# در baseline_models/ اجرا کنید
from env import WebEnv
from agent import Agent  # یا agent سفارشی خودتان

# ساخت محیط با split مشخص
env = WebEnv(
    args,
    split='test',   # 'train', 'eval', یا 'test'
    id=0
)

# حلقه تست
state = env.reset()
done = False
while not done:
    action = my_agent.act(state, env.get_available_actions())
    state, reward, done, info = env.step(action)
```

<div dir="rtl">

### روش ۳: اتصال LLM به عنوان agent

برای اتصال یک LLM (مثل GPT-4 یا Claude) به این محیط:

</div>

```python
import anthropic  # یا openai

client = anthropic.Anthropic()

def llm_agent(observation: str, available_actions: dict) -> str:
    clickables = available_actions.get('clickables', [])
    has_search = available_actions.get('has_search_bar', False)
    
    prompt = f"""You are shopping for a product. Current page:
{observation}

Available actions:
- {"search[query] - search for products" if has_search else ""}
- click options: {clickables[:20]}  

What action do you take? Reply with only the action string."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

# سپس در حلقه ارزیابی استفاده کنید
```

<div dir="rtl">

### نکات مهم برای تست

۱. **شماره goal ها**: goals 0-499 برای test، 500-1499 برای eval، 1500+ برای train
۲. **حداکثر steps**: معمولاً ۱۵ تا ۳۰ step per episode کافی است
۳. **معیار موفقیت**: reward=1.0 یعنی موفقیت کامل
۴. **سرور**: برای حالت متنی نیاز به Flask server ندارید؛ SimServer همه چیز را درون Python شبیه‌سازی می‌کند

</div>

---

## داده‌ها: حجم و مکان

<div dir="rtl">

### داده‌های آماده در مخزن

</div>

```
baseline_models/data/
├── human_goals.json          (1.01 MB)  — 6,910 دستورالعمل انسانی
├── items_human_ins.json      (5.14 MB)  — نگاشت محصول ← دستورالعمل‌ها
├── goal_query_map.json       (250 KB)   — نگاشت دستی دستورالعمل ← query جستجو
├── goal_query_predict.json   (7.26 MB)  — top-10 query تولیدشده توسط BART
└── il_trajs_finalized_images.zip (19.66 MB) — trajectory های IL برای آموزش
```

<div dir="rtl">

### داده‌هایی که باید دانلود شوند (از طریق setup.sh)

</div>

| فایل | حجم تقریبی | توضیح |
|------|------------|-------|
| `items_shuffle_1000.json` | ~50MB | ۱,۰۰۰ محصول نمونه |
| `items_shuffle.json` | ~5GB | تمام ۱.۱۸M محصول |
| `il_trajs_finalized_images.jsonl` | ~2GB | trajectory های کامل با تصویر |
| `search_engine/indexes/` | ~500MB | ایندکس‌های Lucene |

<div dir="rtl">

### ساختار یک محصول

</div>

```json
{
  "asin": "B07XYZABC",
  "name": "Men's Running Shoes",
  "Title": "Nike Air Zoom Running Shoes",
  "category": "Shoes",
  "product_category": "Clothing > Shoes > Running",
  "query": "running shoes men",
  "Description": ["Lightweight mesh upper...", "..."],
  "BulletPoints": ["True to size", "Machine washable"],
  "Reviews": [{"rating": 5, "review": "Great shoes!"}],
  "Rating": 4.3,
  "MainImage": "https://...",
  "pricing": [49.99, 89.99],
  "customization_options": {
    "Size": {"7": "B07XYZ001", "8": "B07XYZ002"},
    "Color": {"Black": "B07XYZ003", "White": "B07XYZ004"}
  },
  "options": {"size": ["7", "8", "9"], "color": ["black", "white"]}
}
```

<div dir="rtl">

### ساختار یک هدف (goal)

</div>

```json
{
  "asin": "B07XYZABC",
  "instruction_text": "Find a pair of men's running shoes in size 8, black color, under $70",
  "attributes": ["running", "men"],
  "goal_options": {"size": "8", "color": "black"},
  "price_upper": 70.0,
  "category": "Shoes",
  "query": "running shoes",
  "name": "Nike Air Zoom...",
  "weight": 1.0
}
```

---

## نسخه‌های مختلف پروژه

<div dir="rtl">

### نسخه‌های اندازه داده

پروژه سه سطح داده دارد که می‌توان در `web_agent_site/utils.py` تنظیم کرد:

</div>

| نسخه | تعداد محصول | کاربرد |
|------|------------|---------|
| Small (100) | 100 | توسعه و debug سریع |
| Medium (1K) | 1,000 | آموزش و ارزیابی استاندارد |
| Full (1.18M) | 1,180,000 | مقیاس کامل |

<div dir="rtl">

### نسخه‌های محیط (Gym IDs)

</div>

```python
# در web_agent_site/envs/__init__.py ثبت شده‌اند:
'WebAgentTextEnv-v0'   # حالت متنی — برای ML
'WebAgentSiteEnv-v0'   # حالت HTML با Selenium
```

<div dir="rtl">

### نسخه‌های مدل پایه

۱. **Rule-based**: قوانین دستی بدون یادگیری (baseline ضعیف)
۲. **IL (Imitation Learning)**: BERT + BART آموزش‌دیده روی trajectory های انسانی
۳. **IL + RL**: Fine-tuning با Reinforcement Learning بعد از IL

</div>

---

## فرایند ارزیابی

<div dir="rtl">

### تقسیم‌بندی داده

</div>

| Split | Goal های | تعداد |
|-------|---------|-------|
| Test | 0 – 499 | 500 |
| Eval | 500 – 1499 | 1000 |
| Train | 1500+ | ~5410 |

<div dir="rtl">

### معیارهای ارزیابی

#### پاداش نرمال‌شده (Normalized Reward)
مقیاس 0 تا 1 که نشان می‌دهد agent چه نسبتی از هدف را برآورده کرده:
</div>

```
r_final = (r_att × w_att + r_option × w_option + r_price × w_price) × r_type

w_att    = num_attributes / (num_attributes + num_options + 1)
w_option = num_options    / (num_attributes + num_options + 1)
w_price  = 1              / (num_attributes + num_options + 1)
```

<div dir="rtl">

#### Harsh Reward (موفقیت کامل)
بایناری: reward=1.0 یعنی موفقیت، هر عدد کمتر یعنی شکست.

#### اجرای ارزیابی

</div>

```bash
cd baseline_models
python test.py \
    --bart_ckpt ./ckpts/web_search/checkpoint-800 \
    --bert_ckpt ./ckpts/web_click/epoch_9/model.pth \
    --split test \
    --num_goals 500
```

<div dir="rtl">

### نتایج مرجع (از مقاله)

</div>

| روش | Score (%) | SR (%) |
|-----|-----------|--------|
| Random | 9.6 | 0.0 |
| Rule-based | 45.3 | 5.4 |
| IL (BERT+BART) | 59.9 | 29.1 |
| IL + RL | 62.4 | 28.7 |
| Human | 82.1 | 59.6 |

---

## نحوه تولید داده

<div dir="rtl">

### داده محصولات

محصولات از آمازون با web scraping جمع‌آوری شده‌اند. فیلدهایی مثل عنوان، توضیحات، صفات، گزینه‌ها، قیمت، و تصویر استخراج شده‌اند. ASIN (Amazon Standard Identification Number) شناسه منحصربه‌فرد هر محصول است.

### دستورالعمل‌های انسانی (Human Instructions)

۱. **کارگران MTurk** با محصولات واقعی وب‌سایت تعامل داشتند
۲. از آن‌ها خواسته شد دستورالعمل خرید برای یک محصول بنویسند (به فارسی تقریبی: "یک کفش ورزشی مردانه سایز 8 به رنگ مشکی زیر 70 دلار پیدا کن")
۳. دستورالعمل‌ها با محصول هدف زوج شدند تا Ground Truth ایجاد شود
۴. صفات و گزینه‌های الزامی به صورت structured annotate شدند

### Trajectory های آموزش (IL Data)

۱. کارگران MTurk مراحل واقعی خرید (کلیک‌ها و جستجوها) را ثبت کردند
۲. هر trajectory شامل: [state, available_actions, action_chosen, reward] در هر step
۳. Feature های تصویری با ResNet از تصاویر محصول استخراج شد
۴. ذخیره‌شده در: `il_trajs_finalized_images.jsonl`

### Query های جستجو

- **دستی** (`goal_query_map.json`): annotators انسانی query مناسب برای هر هدف نوشتند
- **خودکار** (`goal_query_predict.json`): مدل BART top-10 query پیشنهادی تولید کرد

</div>

---

## اندازه‌گیری عملکرد مدل و Ground Truth

<div dir="rtl">

### Ground Truth چیست؟

Ground Truth در این پروژه **چندلایه** است:

#### لایه ۱: محصول هدف
هر goal دقیقاً یک `asin` (شناسه محصول) مشخص دارد. اگر agent همان محصول را خریداری کند، بخشی از reward را می‌گیرد.

#### لایه ۲: صفات الزامی
لیست `attributes` در goal مشخص می‌کند چه خصوصیاتی باید در محصول باشد.  
مثال: `["waterproof", "lightweight"]`

#### لایه ۳: گزینه‌های سفارشی
`goal_options` مشخص می‌کند چه سایز/رنگی باید انتخاب شود.  
مثال: `{"size": "8", "color": "black"}`

#### لایه ۴: محدودیت قیمت
`price_upper` حداکثر قیمت مجاز را تعیین می‌کند.

### نحوه محاسبه عملکرد

</div>

```python
# از web_agent_site/engine/goal.py

def get_reward(purchased_product, goal, price, options, **kwargs):
    # تطابق نوع محصول (دسته و عنوان)
    r_type = get_type_reward(purchased_product, goal)
    
    # تطابق صفات
    goal_attrs = set(goal['attributes'])
    purchased_attrs = set(purchased_product.get('Attributes', []))
    r_att = len(goal_attrs & purchased_attrs) / len(goal_attrs) if goal_attrs else 1.0
    
    # تطابق گزینه‌های سفارشی
    goal_opts = goal['goal_options']
    r_option = sum(
        1 for k, v in goal_opts.items() 
        if options.get(k) == v
    ) / len(goal_opts) if goal_opts else 1.0
    
    # قیمت
    r_price = 1.0 if price <= goal['price_upper'] else 0.0
    
    # وزن‌ها
    n_att = len(goal_attrs)
    n_opt = len(goal_opts)
    total = n_att + n_opt + 1
    
    reward = (
        r_att * (n_att / total) +
        r_option * (n_opt / total) +
        r_price * (1 / total)
    ) * r_type
    
    return reward  # عدد بین 0.0 و 1.0
```

<div dir="rtl">

### اجرای ارزیابی سفارشی

</div>

```python
# برای اندازه‌گیری agent خودتان
scores = []
successes = 0

for goal_idx in range(500):  # test set
    env.reset(goal_idx=goal_idx)
    # اجرای agent...
    final_reward = env.get_reward()
    scores.append(final_reward)
    if final_reward == 1.0:
        successes += 1

print(f"Average Score: {sum(scores)/len(scores)*100:.1f}%")
print(f"Success Rate: {successes/len(scores)*100:.1f}%")
```

---

## ارتباط بین کدها

<div dir="rtl">

### نمودار وابستگی

</div>

```
setup.sh
   └── دانلود داده‌ها → data/items_shuffle_1000.json
   └── ساخت ایندکس → search_engine/indexes/

web_agent_site/engine/engine.py
   ├── بارگذاری از: data/items_shuffle_1000.json
   ├── جستجو از: search_engine/lucene_searcher.py
   ├── پاداش از: goal.py
   └── نرمال‌سازی از: normalize.py

web_agent_site/envs/web_agent_text_env.py
   ├── استفاده از: engine.py (SimServer)
   └── پیاده‌سازی: gym.Env interface

baseline_models/env.py
   └── Wrapper روی: WebAgentTextEnv

baseline_models/agent.py
   ├── استفاده از: models/bert.py (انتخاب action)
   └── استفاده از: models/bart (تولید query)

baseline_models/train_choice_il.py
   ├── خواندن از: data/il_trajs_finalized_images.jsonl
   ├── استفاده از: models/bert.py
   └── ذخیره در: ckpts/web_click/

baseline_models/test.py
   ├── بارگذاری از: ckpts/web_click/ و ckpts/web_search/
   ├── استفاده از: baseline_models/env.py
   └── استفاده از: baseline_models/agent.py
```

<div dir="rtl">

### نقاط اتصال کلیدی

- **محیط ↔ Agent**: از طریق `env.step(action)` و `env.get_available_actions()`
- **Engine ↔ Search**: از طریق `lucene_searcher.py`  
- **Goal ↔ Reward**: `goal.py` با ASIN خریداری‌شده و گزینه‌های انتخاب‌شده محاسبه می‌کند
- **Train ↔ Test Split**: با `goal_idx` range در `env.py` کنترل می‌شود

</div>

---

## مدل‌ها: API یا محلی

<div dir="rtl">

### مدل‌های پایه موجود در پروژه: همه محلی

کد پایه WebShop **هیچ API فراخوانی** به سرویس‌های خارجی ندارد. همه مدل‌ها محلی PyTorch هستند:

</div>

| مدل | نوع | فریمورک | اندازه |
|-----|-----|---------|--------|
| BERT-base-uncased | انتخاب action | HuggingFace | ~440MB |
| BART-large | تولید query | HuggingFace | ~1.6GB |
| RCDQN (RNN) | جایگزین BERT | PyTorch خالص | ~50MB |
| ResNet features | ویژگی تصویر | torchvision | optional |

<div dir="rtl">

### افزودن LLM از API (شامل Claude, GPT-4, Gemini)

پروژه اصلی از API استفاده نمی‌کند، اما معماری آن برای اتصال LLM های API-based بسیار مناسب است. توضیح کامل در بخش بعدی آمده است.

</div>

---

## ادغام یک agent با منطق کد متفاوت

<div dir="rtl">

### چالش اصلی

اگر agent شما با منطق کاملاً متفاوتی نوشته شده (مثلاً یک LLM agent، یک ReAct agent، یک chain-of-thought agent، یا یک RL agent با معماری خاص) باید یک **adapter layer** بسازید.

### سناریو ۱: LLM Agent با prompt engineering

</div>

```python
# llm_webshop_agent.py
import gym
import web_agent_site
import anthropic  # یا openai، google.generativeai، ...

class LLMWebShopAgent:
    def __init__(self, model_name: str):
        self.client = anthropic.Anthropic()
        self.model = model_name
        self.history = []
    
    def act(self, observation: str, available_actions: dict) -> str:
        clickables = available_actions.get('clickables', [])
        has_search = available_actions.get('has_search_bar', False)
        
        # ساخت prompt
        action_desc = []
        if has_search:
            action_desc.append("search[your query]")
        for c in clickables[:15]:
            action_desc.append(f"click[{c}]")
        
        messages = self.history + [{
            "role": "user",
            "content": f"Page:\n{observation}\n\nActions:\n" + "\n".join(action_desc) + "\n\nAction:"
        }]
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=50,
            system="You are a shopping agent. Reply with exactly one action.",
            messages=messages
        )
        
        action = response.content[0].text.strip()
        self.history.append({"role": "user", "content": messages[-1]["content"]})
        self.history.append({"role": "assistant", "content": action})
        return action
    
    def reset(self):
        self.history = []

# ارزیابی
env = gym.make('WebAgentTextEnv-v0', observation_mode='text_rich', num_products=1000)
agent = LLMWebShopAgent("claude-sonnet-4-6")

scores = []
for i in range(500):
    obs = env.reset()
    agent.reset()
    done = False
    while not done:
        available = env.get_available_actions()
        action = agent.act(obs, available)
        obs, reward, done, info = env.step(action)
    scores.append(reward)
    print(f"Episode {i}: reward={reward:.3f}")

print(f"Mean Score: {sum(scores)/len(scores)*100:.1f}%")
```

<div dir="rtl">

### سناریو ۲: ReAct Agent (Reasoning + Acting)

</div>

```python
# react_webshop_agent.py

SYSTEM_PROMPT = """You are a shopping agent using ReAct framework.
For each step:
Thought: [reasoning about current state and goal]
Action: [one of: search[query] or click[option]]

Instructions format: search[query] or click[text]"""

class ReActAgent:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.trajectory = []
        self.goal = None
    
    def set_goal(self, goal_text: str):
        self.goal = goal_text
        self.trajectory = []
    
    def act(self, obs: str, available_actions: dict) -> str:
        self.trajectory.append(f"Observation: {obs[:500]}")
        
        prompt = f"Goal: {self.goal}\n\n" + "\n".join(self.trajectory[-10:])
        
        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = response.content[0].text
        # استخراج Action از متن
        import re
        match = re.search(r'Action:\s*(.+)', text)
        action = match.group(1).strip() if match else "click[Back to Search]"
        
        self.trajectory.append(f"Thought: {text}")
        self.trajectory.append(f"Action: {action}")
        return action
```

<div dir="rtl">

### سناریو ۳: agent با معماری کاملاً متفاوت (مثلاً Transformer-based RL)

اگر مدل شما یک شبکه عصبی با معماری سفارشی است:

</div>

```python
class CustomNNAgent:
    def __init__(self, model_path: str):
        import torch
        self.model = torch.load(model_path)
        self.tokenizer = ...  # tokenizer مدل شما
    
    def encode_state(self, obs: str, clickables: list) -> dict:
        """تبدیل observation به فرمت ورودی مدل شما"""
        tokens = self.tokenizer(obs, return_tensors='pt', truncation=True, max_length=512)
        action_tokens = [
            self.tokenizer(a, return_tensors='pt', truncation=True, max_length=128)
            for a in clickables
        ]
        return {'state_tokens': tokens, 'action_tokens': action_tokens}
    
    def act(self, obs: str, available_actions: dict) -> str:
        clickables = available_actions.get('clickables', [])
        has_search = available_actions.get('has_search_bar', False)
        
        if has_search:
            # اگر مدل شما search query تولید می‌کند
            query = self.model.generate_query(obs)
            return f"search[{query}]"
        
        # encode و score گرفتن برای هر action
        encoded = self.encode_state(obs, clickables)
        with torch.no_grad():
            scores = self.model(**encoded)
        
        best_idx = scores.argmax().item()
        return f"click[{clickables[best_idx]}]"
```

<div dir="rtl">

### نکته کلیدی برای ادغام

تنها چیزی که باید پیاده‌سازی کنید:

</div>

```
observation (str) + available_actions (dict) → action (str)
```

<div dir="rtl">

این یک رابط بسیار ساده است و هر agent با هر معماری را می‌توان به این محیط وصل کرد.

</div>

---

## بومی‌سازی داده

<div dir="rtl">

### چالش‌های بومی‌سازی

پروژه WebShop به طور کامل به زبان انگلیسی است. بومی‌سازی به فارسی یا هر زبان دیگری نیازمند چند مرحله است:

### مرحله ۱: ترجمه دستورالعمل‌ها

</div>

```python
# translate_goals.py
import json
import anthropic

client = anthropic.Anthropic()

def translate_instruction(text: str, target_lang: str = "Persian") -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"Translate this shopping instruction to {target_lang}. Keep product specs (size, color, price) as numbers:\n{text}"
        }]
    )
    return response.content[0].text.strip()

# بارگذاری و ترجمه
with open('baseline_models/data/human_goals.json') as f:
    goals = json.load(f)

fa_goals = []
for goal in goals:
    fa_goal = goal.copy()
    fa_goal['instruction_text_fa'] = translate_instruction(goal['instruction_text'])
    fa_goals.append(fa_goal)

with open('baseline_models/data/human_goals_fa.json', 'w', encoding='utf-8') as f:
    json.dump(fa_goals, f, ensure_ascii=False, indent=2)
```

<div dir="rtl">

### مرحله ۲: ترجمه اطلاعات محصولات

</div>

```python
# translate_products.py — برای items_shuffle_1000.json
fields_to_translate = ['name', 'Title', 'Description', 'BulletPoints']

def translate_product(product: dict) -> dict:
    translated = product.copy()
    for field in fields_to_translate:
        if field in product and product[field]:
            if isinstance(product[field], list):
                translated[f'{field}_fa'] = [translate_instruction(t) for t in product[field][:3]]
            else:
                translated[f'{field}_fa'] = translate_instruction(str(product[field]))
    return translated
```

<div dir="rtl">

### مرحله ۳: بومی‌سازی normalize.py

فایل `web_agent_site/engine/normalize.py` لیست رنگ‌ها و اندازه‌ها را به انگلیسی دارد. برای کارکرد درست با محتوای فارسی باید معادل‌های فارسی اضافه شوند:

</div>

```python
# در normalize.py اضافه کنید:
COLOR_MAP_FA = {
    "مشکی": "black", "سفید": "white", "قرمز": "red",
    "آبی": "blue", "سبز": "green", "زرد": "yellow",
    "قهوه‌ای": "brown", "خاکستری": "gray", "نارنجی": "orange",
    "صورتی": "pink", "بنفش": "purple", "طلایی": "gold"
}

SIZE_MAP_FA = {
    "کوچک": "small", "متوسط": "medium", "بزرگ": "large",
    "خیلی بزرگ": "x-large", "خیلی کوچک": "x-small"
}
```

<div dir="rtl">

### مرحله ۴: ادغام مدل زبانی فارسی‌دان

برای ارزیابی مدل‌های فارسی می‌توانید از مدل‌هایی مثل Claude که فارسی می‌فهمند یا مدل‌های اختصاصی فارسی استفاده کنید. دستورالعمل‌های ترجمه‌شده را به عنوان ورودی goal بگذارید.

### مرحله ۵: بومی‌سازی محصولات

برای یک محیط کاملاً بومی، می‌توانید از محصولات ایرانی (مثلاً دیجی‌کالا) استفاده کنید:
- scraping از دیجی‌کالا یا ترب با رعایت `robots.txt`
- تبدیل به فرمت JSON مطابق ساختار بالا
- ساخت ایندکس Lucene جدید با متن فارسی (Lucene از فارسی پشتیبانی می‌کند)
- نوشتن دستورالعمل‌های خرید فارسی برای محصولات

</div>

---

## نکات ارزشمند اضافی

<div dir="rtl">

### ۱. نقطه ضعف اصلی: Search Query Generation
بزرگ‌ترین گلوگاه عملکرد agent، تولید query جستجوی مناسب است. اگر agent از مرحله جستجو به محصول مناسب نرسد، بقیه مراحل بی‌فایده است. پیشنهاد: از LLM برای این بخش استفاده کنید.

### ۲. حالت‌های مشاهده و تأثیر بر عملکرد
آزمایش کنید که کدام `observation_mode` برای agent شما بهتر است:
- `'text_rich'` معمولاً بهترین توازن اطلاعات/نویز را دارد
- `'html'` اطلاعات بیشتری دارد اما پردازش سخت‌تر است
- `'text'` برای LLM هایی که context window کوچکتری دارند مناسب‌تر است

### ۳. مشکل pagination
agent باید یاد بگیرد که اگر محصول مناسب در صفحه اول نبود، با `click[Next >]` صفحات بعدی را بررسی کند. بسیاری از مدل‌ها این مرحله را نادیده می‌گیرند.

### ۴. Option Selection مشکل‌ساز
رنگ‌ها و سایزها اغلب با نرمال‌سازی می‌توان متطابق کرد، اما مدل باید بفهمد که قبل از `click[Buy Now]` باید تمام گزینه‌های الزامی را انتخاب کند.

### ۵. اندازه context
مشاهدات می‌توانند تا ۱۰۰۰+ توکن باشند. برای LLM های با context window محدود، پیاده‌سازی summarization یا chunking لازم است.

### ۶. مقایسه با بنچمارک‌های جدیدتر
WebShop یکی از اولین benchmarkهای این حوزه بود. بنچمارک‌های جدیدتر مثل **Mind2Web** و **WebArena** مشکلات واقعی‌تری دارند، اما WebShop هنوز پرکاربردترین در مقالات است چون محیط کنترل‌شده و معیار ارزیابی دقیق دارد.

### ۷. ارتباط با Sim-to-Real Transfer
پوشه `transfer/` نشان می‌دهد می‌توان همان agent آموزش‌دیده روی WebShop را روی آمازون واقعی تست کرد. این یک مزیت بزرگ برای پایان‌نامه است — می‌توان ادعا کرد مدل قابلیت تعمیم به محیط واقعی دارد.

### ۸. محاسبه پاداش به عنوان metric تشخیصی
اجزای پاداش (`r_type`, `r_att`, `r_option`, `r_price`) به جای یک عدد واحد، اطلاعات تشخیصی دقیق می‌دهند:
- `r_type` پایین → agent محصول اشتباهی خریده
- `r_att` پایین → محصول صفات لازم را ندارد  
- `r_option` پایین → گزینه‌های سفارشی درست انتخاب نشده
- `r_price` پایین → قیمت از بودجه بیشتر است

### ۹. یکپارچه‌سازی WandB
سیستم لاگ‌گذاری با WandB یکپارچه است. برای مقایسه agent های مختلف خودتان، از آن استفاده کنید:

</div>

```python
import wandb
wandb.init(project="webshop-my-agent", name="llm-react-v1")
wandb.log({"score": reward, "episode": i, "success": reward == 1.0})
```

<div dir="rtl">

### ۱۰. چکیده پیشنهادات برای پایان‌نامه

اگر هدف ارزیابی یک agent جدید است، پیشنهاد ترتیب کار:

۱. محیط را با ۱۰۰۰ محصول راه‌اندازی کنید
۲. ابتدا agent RandomPolicy را اجرا کنید تا baseline داشته باشید
۳. agent خودتان را از طریق `WebAgentTextEnv-v0` وصل کنید
۴. روی ۵۰۰ goal اول (test split) ارزیابی کنید
۵. نتایج را با جدول مقاله اصلی مقایسه کنید
۶. اجزای پاداش را جداگانه گزارش دهید
۷. برای بومی‌سازی، دستورالعمل‌ها را به فارسی ترجمه کنید و یک LLM فارسی‌دان را مقایسه کنید

</div>

---

<div dir="rtl">

*این سند توسط Claude Code تهیه شده است — تاریخ: ۱۴۰۵/۰۴/۰۵*

</div>
