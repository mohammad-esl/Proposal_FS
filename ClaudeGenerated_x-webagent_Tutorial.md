# X-WebAgentBench — راهنمای کامل پروژه

<div dir="rtl">

## معرفی کلی

**X-WebAgentBench** یک بنچمارک تعاملی و چندزبانه برای ارزیابی agent‌های مبتنی بر LLM است که در محیط وب اجرا می‌شوند. این پروژه در **ACL 2025 Findings** منتشر شده است.

محیط شبیه‌سازی‌شده یک فروشگاه آنلاین است. agent باید دستورالعمل خرید را بفهمد، محصول مناسب را جستجو کند، صفحات آن را مرور کند، گزینه‌های درست را انتخاب کند و خرید را نهایی کند.

### زبان‌های پشتیبانی‌شده (۱۵ زبان)

</div>

| کد | زبان | کد | زبان | کد | زبان |
|----|------|----|------|----|------|
| `en` | English | `ru` | Russian | `hi` | Hindi |
| `zh` | Chinese | `tr` | Turkish | `sw` | Swahili |
| `fr` | French | `ar` | Arabic | `ur` | Urdu |
| `es` | Spanish | `vi` | Vietnamese | `el` | Greek |
| `de` | German | `th` | Thai | `bg` | Bulgarian |

---

<div dir="rtl">

## ساختار فایل‌ها

</div>

```
X-WebAgentBench/
├── README.md
├── requirements.txt
├── eval_environment.yml             # محیط conda برای ارزیابی
├── xwebagentbench_environment.yml   # محیط conda برای سرور
├── reset_index.sh                   # ساخت ایندکس‌های Lucene برای همه زبان‌ها
│
├── Eval/
│   ├── main.py                      # نقطه ورودی اصلی
│   ├── model.py                     # class model: مدل‌های API-based
│   ├── os_model.py                  # class os_model: مدل‌های محلی
│   ├── WebShopEnv.py                # HTTP client + HTML parser
│   ├── WebShop_test.py              # حلقه تست (چهار روش)
│   ├── WebShop_prompt.py            # قالب‌های prompt
│   ├── multilingual_text.py         # متون و action‌های چندزبانه
│   ├── re_eval.py                   # ارزیابی مجدد لاگ‌های موجود
│   └── split_paragraph.py           # تقسیم متن بلند برای Google Translate
│
├── web_agent_site/
│   ├── app_multi.py                 # Flask server
│   ├── utils.py                     # مسیرها، logger، generate_mturk_code
│   ├── multilingual_text.py         # متن رابط کاربری برای ۱۵ زبان
│   ├── engine/
│   │   ├── engine.py                # جستجو، بارگذاری محصول، رندر HTML
│   │   ├── goal.py                  # محاسبه reward
│   │   └── normalize.py             # نرمال‌سازی رنگ و سایز
│   └── templates/
│       ├── search_page.html
│       ├── results_page.html
│       ├── item_page.html
│       ├── description_page.html
│       ├── features_page.html
│       ├── review_page.html
│       ├── attributes_page.html
│       └── done_page.html
│
├── search_engine/
│   ├── lucene_searcher.py           # اسکریپت تست ساده (نه wrapper)
│   ├── convert_product_file_format.py
│   └── run_indexing.sh
│
├── data/
│   ├── items_shuffle/               # items_shuffle_40k_{lang}.json
│   ├── items_ins/                   # items_ins_v2_40k_{lang}.json
│   ├── items_human_ins/             # items_human_ins_{lang}.json
│   └── reviews.json
│
└── user_session_logs/
    ├── all_trajs/
    └── mturk/
```

---

<div dir="rtl">

## معماری کلی

</div>

```
Eval/main.py
    ↓ انتخاب مدل و روش
Eval/WebShop_test.py
    ↓ حلقه تست
Eval/WebShopEnv.py  ──HTTP──→  web_agent_site/app_multi.py
                                    ↓
                              engine/engine.py   (جستجو + HTML)
                              engine/goal.py     (امتیاز)
```

---

<div dir="rtl">

## توضیح دقیق هر فایل کد

### `Eval/main.py`

آرگومان‌های CLI:

</div>

```python
# --model   : 'gpt-3.5-turbo' | 'gpt-4o' | 'deepseek_v2' | 'deepseek-reasoner'
#             'qwq' | 'qwen2' | 'mistral' | 'llama3'
# --method  : 'direct' | 'translate_en' | 'self-translate_en' | 'clp_en'
# --language: 'zh' | 'fr' | 'es' | 'de' | 'el' | 'bg' | 'ru' | 'tr'
#             'ar' | 'vi' | 'th' | 'hi' | 'sw' | 'ur'  (یا 'en')
# --test_n  : تعداد نمونه (پیش‌فرض 200)
# --device  : پیش‌فرض 'cuda:0'
# --root_log_path: پیش‌فرض './saved_log'
```

<div dir="rtl">

**نکته مهم از کد**: `deepseek_v2` در `main.py` به `model.deepseek_v2_generator` ارجاع می‌دهد اما این تابع در `model.py` **وجود ندارد** — باگ موجود در ریپو است. همچنین `llama3` پیاده‌سازی نشده؛ فقط یک پیام پرینت می‌شود:

</div>

```python
elif args.model == 'llama3':
    print('Please follow the official tutorial and modify our code in os_model.py, '
          'or you can use API to call the llama series model (we use deepinfra).')
```

---

<div dir="rtl">

### `Eval/model.py`

generatorها **static method داخل `class model()`** هستند، نه تابع مستقل:

</div>

```python
class model():
    def gpt_generator(text, history=[], language='en', mode='direct'):
        ...
    def gpt4_generator(text, history=[], language='en', mode='direct'):
        ...
    def deepseekreasoner_generator(text, history=[], language='en', mode='direct'):
        ...
    def qwq_generator(text, history=[], language='en', mode='direct'):
        ...
```

<div dir="rtl">

فراخوانی در `main.py`:

</div>

```python
test_model = WebShop_test(model.gpt_generator, saved_log_path)
test_model = WebShop_test(model.gpt4_generator, saved_log_path)
test_model = WebShop_test(model.deepseekreasoner_generator, saved_log_path)
test_model = WebShop_test(model.qwq_generator, saved_log_path)
```

<div dir="rtl">

جزئیات هر generator:

</div>

| تابع | مدل دقیق | سرویس | base_url |
|------|----------|--------|----------|
| `gpt_generator` | `gpt-3.5-turbo` | `client_gpt` | تنظیم دستی |
| `gpt4_generator` | `gpt-4o` | `client_gpt` | تنظیم دستی |
| `deepseekreasoner_generator` | `deepseek-reasoner` | `client_deepseek` | `https://api.deepseek.com/v1` |
| `qwq_generator` | `Qwen/QwQ-32B-Preview` | `client_deepinfra` | `https://api.deepinfra.com/v1/openai` |

<div dir="rtl">

همه از OpenAI SDK با `base_url` متفاوت استفاده می‌کنند. `BREAK_TIMES_LIMIT = 5` بار تلاش مجدد در صورت خطا.

**نکته**: در `gpt_generator` و `gpt4_generator` و `qwq_generator`، پاسخ مدل با role `"system"` به history اضافه می‌شود (نه `"assistant"`). در `deepseekreasoner_generator` و `mistral_generator` با role `"assistant"` اضافه می‌شود.

</div>

---

<div dir="rtl">

### `Eval/os_model.py`

کلاس `os_model` مدل‌های محلی را بارگذاری می‌کند:

</div>

```python
class os_model():
    def __init__(self, model, device):
        if model == 'qwen2':
            model_name = "Qwen/Qwen2-7B-Instruct"
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif model == 'mistral':
            self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map=device)
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
```

<div dir="rtl">

تفاوت مهم در Mistral: prompt سیستم را به عنوان `"user"` با پاسخ ثابت `"OK"` از `"assistant"` می‌فرستد (چون Mistral system role ندارد). خروجی با `split('[/INST]')[-1]` پردازش می‌شود.

</div>

---

<div dir="rtl">

### `Eval/WebShop_prompt.py`

**فرمت پاسخ مورد انتظار از agent** (از `get_base_prompt`):

</div>

```
Thought: I think ...
Action: click[<something>]
```

<div dir="rtl">

`get_action()` در `WebShop_test.py` با `action.split('Action: ')[1]` اقدام را استخراج می‌کند.

action‌ها **زبان‌اختصاصی** هستند — از `env_search[language]` و `env_click[language]` در `multilingual_text.py`:

</div>

```python
# انگلیسی:    search[keyword]        click[Buy Now]
# چینی:       搜索[keyword]           点击[现在购买]
# فرانسوی:    recherche[keyword]      cliquez sur[Acheter maintenant]
```

<div dir="rtl">

توابع موجود در این فایل:

</div>

| تابع | کاربرد |
|------|--------|
| `get_base_prompt(language)` | prompt اصلی با تعریف action‌ها |
| `get_trans_prompt_en(text, language)` | ترجمه متن به انگلیسی |
| `get_trans_prompt(text, language)` | ترجمه متن به زبان هدف |
| `get_clp_base_prompt(language)` | prompt مرحله اول CLP |
| `get_clp_1_prompt(observation, language)` | درخواست توضیح step-by-step |
| `get_clp_2_prompt(prev_action, language)` | درخواست action بعد از alignment |

---

<div dir="rtl">

### `Eval/WebShopEnv.py`

آدرس سرور در خط ۸ تنظیم می‌شود:

</div>

```python
WEBSHOP_URL = "webshop url"   # باید با IP سرور جایگزین شود
```

<div dir="rtl">

تابع `webshop_text()` HTML را با BeautifulSoup parse می‌کند:

</div>

```python
# <button>          →  \n[متن دکمه]
# <label>           →  [option]  یا  [[option]] اگر در URL باشد (= انتخاب‌شده)
# class=product-link→  \n[ASIN]
# متن عادی         →  \n text
```

<div dir="rtl">

بازگشتی: `(observation_str, info_dict)` که `info_dict` شامل `option_types`، `asins`، و `reward` است.

کلاس `webshopEnv` یک state machine است:
- `'reset'` → `page_type = 'init'`
- `'search'` → `page_type = 'search'`
- `'click'` → بسته به دکمه: `'item'`, `'item_sub'`, `'end'`, یا برگشت

**محدودیت صفحه**: صفحه بیشتر از ۱۰ خطا می‌دهد (`assert False`).

</div>

---

<div dir="rtl">

### `Eval/WebShop_test.py`

کلاس `WebShop_test` چهار متد تست دارد:

</div>

| متد | شرح |
|-----|-----|
| `test_direct` | مستقیم با زبان اصلی |
| `test_translate_en` | Google Translate صفحات را به انگلیسی برمی‌گرداند |
| `test_self_translate_en` | خود LLM ترجمه می‌کند (`mode='translate_en'`) |
| `test_clp_en` | دو مرحله: `mode='clp'` برای alignment، سپس action |

<div dir="rtl">

حلقه اصلی هر متد:

</div>

```
reset → مشاهده اولیه → (ترجمه؟) → search → حلقه find/click → buy → reward
```

<div dir="rtl">

محدودیت جستجوی مجدد: `limit_re_search_times=3`. بعد از آن `reward=0`.

نتایج در `{root_log_path}/{model}/{method}/{method}_{language}.json` ذخیره می‌شوند.

</div>

---

<div dir="rtl">

### `Eval/multilingual_text.py`

مهم‌ترین متغیرها که agent باید از آنها استفاده کند:

</div>

```python
env_search  = {'en': 'search',  'zh': '搜索',  'fr': 'recherche', ...}
env_click   = {'en': 'click',   'zh': '点击',   'fr': 'cliquez sur', ...}
webshop_buy = {'en': 'Buy Now', 'zh': '现在购买', 'fr': 'Acheter maintenant', ...}
webshop_prev= {'en': 'Prev',    'zh': '上一页',  ...}
webshop_next= {'en': 'Next',    'zh': '下一页',  ...}
webshop_back_btu = {'en': 'Back to Search', 'zh': '返回搜索页面', ...}
```

---

<div dir="rtl">

### `Eval/re_eval.py`

ارزیابی مجدد لاگ‌های JSON موجود بدون اجرای مجدد agent. مفید برای محاسبه score با معیار تغییرکرده.

### `Eval/split_paragraph.py`

تقسیم متن طولانی به قطعات کمتر از `max_length` کاراکتر. برای دور زدن محدودیت ۵۰۰۰ کاراکتر Google Translate استفاده می‌شود.

</div>

---

<div dir="rtl">

### `web_agent_site/utils.py`

مسیرهای داده و توابع کمکی:

</div>

```python
DEFAULT_FILE_PATH  = '../data/items_shuffle/'
DEFAULT_ATTR_PATH  = '../data/items_ins/'
HUMAN_ATTR_PATH    = '../data/items_human_ins/'
DEFAULT_REVIEW_PATH= '../data/reviews.json'

def generate_mturk_code(session_id: str) -> str:
    sha = hashlib.sha1(session_id.encode())
    return sha.hexdigest()[:10].upper()

def setup_logger(session_id, user_log_dir): ...   # لاگ session به JSONL
def random_idx(cum_weights): ...                  # نمونه‌برداری وزن‌دار
```

---

<div dir="rtl">

### `web_agent_site/engine/engine.py`

ثابت‌های مهم:

</div>

```python
SEARCH_RETURN_N = 50    # حداکثر نتایج جستجو
PRODUCT_WINDOW  = 5     # محصول در هر صفحه
END_BUTTON      = 'Buy Now'
NEXT_PAGE       = 'Next >'
PREV_PAGE       = '< Prev'
BACK_TO_SEARCH  = 'Back to Search'
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features':    'features_page.html',
    'Reviews':     'review_page.html',
    'Attributes':  'attributes_page.html',
}
```

<div dir="rtl">

توابع کلیدی:

**`load_products(filepath, attr_path, human_attr_path, num_products=None, human_goals=True)`**
فایل‌های JSON را می‌خواند و ساختار محصول را می‌سازد. قیمت با `generate_product_prices` تولید می‌شود.

**`generate_product_prices(all_products)`**
قیمت هر محصول با `random.uniform(*pricing[:2])` از بازه قیمت تصادفی تولید می‌شود — پس بین اجراها فرق دارد.

**`init_search_engine(language, num_products=None)`**
`LuceneSearcher` را از پوشه `search_engine/{language}/{indexes}` بارگذاری می‌کند:

</div>

```python
# num_products=100    → indexes_100
# num_products=1000   → indexes_1k
# num_products=100000 → indexes_100k
# num_products=None   → indexes   (کامل)
```

<div dir="rtl">

**`get_top_n_product_from_keywords(keywords, search_engine, ...)`**
کلیدواژه‌های خاص **اینجا** (در engine.py) پردازش می‌شوند:

</div>

```python
if keywords[0] == '<r>':   # نمونه‌برداری تصادفی 50 محصول
elif keywords[0] == '<a>': # فیلتر بر اساس Attributes
elif keywords[0] == '<c>': # فیلتر بر اساس category
elif keywords[0] == '<q>': # فیلتر بر اساس query
else:                      # جستجوی Lucene BM25
    hits = search_engine.search(keywords, k=SEARCH_RETURN_N)
```

<div dir="rtl">

**`get_product_per_page(top_n_products, page)`**

</div>

```python
return top_n_products[(page - 1) * PRODUCT_WINDOW : page * PRODUCT_WINDOW]
```

---

<div dir="rtl">

### `web_agent_site/engine/goal.py`

جزئیات کامل در بخش «Ground Truth» پایین‌تر.

### `web_agent_site/engine/normalize.py`

**`normalize_color(color_string: str) -> str`**
اولین رنگ یافت‌شده از `COLOR_SET` را برمی‌گرداند. اگر هیچ‌کدام نباشد، رشته اصلی برمی‌گردد.

**`normalize_color_size(product_prices: dict) -> Tuple[dict, dict]`**
ورودی: dict با کلیدهای `(asin, color, size)`. دو mapping برمی‌گرداند: رنگ → مقدار نرمال، سایز → pattern.

`COLOR_SET` شامل رنگ‌های از پیش‌تعریف‌شده است (alabaster تا yellow).
`SIZE_SET` شامل: `'xx-large', '3x-large', '4x-large', '5x-large', 'x-large', 'x-small', 'medium', 'large', 'small', 'queen', 'twin', 'full', 'king', 'one size', 'pack'`
`SIZE_PATTERNS` شامل regex هایی مثل `neck.*sleeve`، `w x.*l`، `.*inch`، `.*mm`، `\d+cm$` و غیره.

### `search_engine/lucene_searcher.py`

این فایل یک **اسکریپت تست** است، نه یک کلاس wrapper:

</div>

```python
searcher = LuceneSearcher('indexes')
hits = searcher.search('rubber sole shoes', k=20)
for hit in hits:
    doc = searcher.doc(hit.docid)
    obj = json.loads(doc.raw())['product']['Title']
    print(obj)
```

<div dir="rtl">

جستجوی واقعی از طریق `LuceneSearcher` مستقیماً در `engine.py::init_search_engine()` انجام می‌شود.

### `search_engine/convert_product_file_format.py`

محصولات JSON را به JSONL قابل ایندکس تبدیل می‌کند. **چهار variant** می‌سازد:

</div>

```python
# فایل ورودی: items_shuffle_40k_{language}.json
doc['contents'] = ' '.join([Title, Description, BulletPoints[0], option_text]).lower()
doc['id'] = p['asin']
doc['product'] = p  # کل شیء محصول

with open(f'./{language}/resources_100/documents.jsonl', 'w+') as f:
    for doc in docs[:100]: ...          # 100 محصول

with open(f'./{language}/resources/documents.jsonl', 'w+') as f:
    for doc in docs: ...                # همه (~40k)

with open(f'./{language}/resources_1k/documents.jsonl', 'w+') as f:
    for doc in docs[:1000]: ...         # 1000 محصول

with open(f'./{language}/resources_100k/documents.jsonl', 'w+') as f:
    for doc in docs[:100000]: ...       # 100k محصول
```

</div>

---

<div dir="rtl">

## نمودار وابستگی فایل‌ها

</div>

```
Eval/main.py
├── from WebShop_test import WebShop_test
├── from model import model
└── from os_model import os_model

Eval/WebShop_test.py
├── from WebShopEnv import webshopEnv
├── from WebShop_prompt import *
├── from multilingual_text import *
├── from deep_translator import GoogleTranslator
└── from split_paragraph import split_paragraph

Eval/WebShopEnv.py
├── from bs4 import BeautifulSoup
├── from multilingual_text import *
└── HTTP GET → web_agent_site/app_multi.py

web_agent_site/engine/engine.py
├── from pyserini.search.lucene import LuceneSearcher
├── from web_agent_site.utils import BASE_DIR, DEFAULT_FILE_PATH, ...
└── reads: data/items_shuffle/items_shuffle_40k_{lang}.json

web_agent_site/engine/goal.py
├── import spacy  (en_core_web_lg)
├── from thefuzz import fuzz
├── from web_agent_site.engine.normalize import normalize_color
└── from web_agent_site.multilingual_text import get_price_prompt

web_agent_site/engine/normalize.py
└── (فقط import re و typing)

Eval/model.py
├── from openai import OpenAI
└── from WebShop_prompt import *

Eval/os_model.py
├── from transformers import AutoModelForCausalLM, AutoTokenizer
└── from WebShop_prompt import *
```

---

<div dir="rtl">

## داده‌ها و variant های مختلف

### نام دقیق فایل‌های داده

</div>

```
data/items_shuffle/items_shuffle_40k_{lang}.json      # محصولات
data/items_ins/items_ins_v2_40k_{lang}.json           # صفات محصولات
data/items_human_ins/items_human_ins_{lang}.json      # دستورالعمل‌های انسانی
data/reviews.json                                     # نظرات (در کد comment شده)
```

<div dir="rtl">

### چهار variant اندازه

</div>

| Variant | تعداد محصول | پوشه resources | پوشه indexes |
|---------|------------|----------------|--------------|
| کوچک | 100 | `resources_100/` | `indexes_100` |
| متوسط | 1,000 | `resources_1k/` | `indexes_1k` |
| کامل | ~40,000 | `resources/` | `indexes` |
| بزرگ | 100,000 | `resources_100k/` | `indexes_100k` |

```python
init_search_engine('en', num_products=100)     # indexes_100
init_search_engine('en', num_products=1000)    # indexes_1k
init_search_engine('en', num_products=100000)  # indexes_100k
init_search_engine('en', num_products=None)    # indexes (کامل)
```

<div dir="rtl">

### split های ارزیابی

این پروژه train/test/eval split جداگانه ندارد:
- هر زبان **۲۰۰ نمونه** انسانی دارد (`fixed_0` تا `fixed_199`)
- همه به عنوان test set استفاده می‌شوند
- `--test_n` مشخص می‌کند چند نمونه ارزیابی شود (پیش‌فرض ۲۰۰)

### ساختار یک محصول در JSON

از خواندن `load_products()` در `engine.py`:

</div>

```python
{
    "asin": "B09743DFJC",
    "category": "...",
    "query": "...",           # lowercase strip شده
    "product_category": "... › ...",
    "name": "...",            # = p['name'] از فایل JSON
    "Title": "...",           # = p['name']
    "Description": "...",     # = p['full_description'] یا '-'
    "BulletPoints": [...],    # = p['small_description']
    "pricing": [39.9, 49.99], # بازه قیمت
    "Price": "$39.9 to $49.99",
    "MainImage": "...",       # = p['images'][0]
    "Rating": "N.A.",         # از reviews (در کد فعلی comment شده)
    "Reviews": [],            # از reviews (در کد فعلی comment شده)
    "Attributes": [...],      # از items_ins فایل
    "options": {              # از customization_options
        "color": ["black", "white"],
        "size": ["s", "m", "l"]
    },
    "option_to_image": {"black": "url", ...},
    "instructions": [...]     # از items_human_ins فایل (اگر human_goals=True)
}
```

---

<div dir="rtl">

## Ground Truth چهارلایه و فرمول دقیق پاداش

هر هدف خرید چهار لایه دارد:

</div>

```
لایه ۱ — نوع محصول (r_type)   : ضریب کلی [0.0, 0.1, 0.5, 1.0]
لایه ۲ — صفات (attributes)    : مثل "waterproof"
لایه ۳ — گزینه‌ها (options)    : مثل color="black", size="L"
لایه ۴ — قیمت (r_price)       : باینری (0 یا 1)
```

<div dir="rtl">

### فرمول دقیق از `goal.py`

</div>

```python
def get_reward(purchased_product, goal, purchased_product_en, goal_en, price, options):

    # لایه ۱: r_type — روی نسخه انگلیسی محصول محاسبه می‌شود (با spaCy)
    r_type_dict = get_type_reward(purchased_product_en, goal_en)
    # r_type = 1.0  اگر query_match OR category_match OR title_score > 0.2
    # r_type = 0.5  اگر هیچکدام نباشد
    # r_type = 0.1  اگر title_score < 0.1
    # r_type = 0.0  اگر title_score == 0.0

    # لایه ۴: price
    r_price = (price <= goal['price_upper']) if goal['price_upper'] > 0 else None

    # لایه ۲: attributes — fuzzy match با آستانه 85% (thefuzz)
    # جستجو در: Attributes، Title، BulletPoints، Description
    r_att, num_attr_matches = get_attribute_reward(purchased_product, goal)

    # لایه ۳: options — fuzzy match با آستانه 85% + normalize_color
    r_option, num_option_matches = get_option_reward(
        list(options.values()),
        goal['goal_options'].items()
    )

    # فرمول نهایی:
    total_reward = (num_attr_matches + num_option_matches + r_price) \
                 / (len(goal['attributes']) + len(goal['goal_options']) + 1)
    total_reward *= r_type_dict['r_type']
    return total_reward  # بازه: [0.0, 1.0]
```

<div dir="rtl">

### نکته مهم: r_type روی نسخه انگلیسی

`get_type_reward` با `purchased_product_en` و `goal_en` فراخوانی می‌شود — نه نسخه زبان اصلی. این به این دلیل است که spaCy برای استخراج noun‌ها نیاز به انگلیسی دارد.

### verbose mode

</div>

```python
reward, info = get_reward(..., verbose=True)
# info شامل: r_type, r_att, r_option, r_price
#             query_match, category_match, title_score
#             w_att, w_option (وزن هر مؤلفه)
```

<div dir="rtl">

### مثال عددی

هدف: ۲ صفت، ۲ گزینه، یک محدودیت قیمت. agent همه را درست انتخاب می‌کند:

</div>

```
total = (2 + 2 + 1) / (2 + 2 + 1) × 1.0 = 1.0
```

<div dir="rtl">

اگر r_type=0.0 (دسته‌بندی کاملاً اشتباه)، همه چیز صفر می‌شود.

### قیمت تصادفی در هر اجرا

`generate_product_prices` با `random.uniform(*pricing[:2])` قیمت می‌سازد، پس بین اجراها فرق می‌کند.

</div>

---

<div dir="rtl">

## نصب و راه‌اندازی

</div>

```bash
# محیط سرور
conda env create -f xwebagentbench_environment.yml
conda activate xwebagentbench

# محیط ارزیابی (جداگانه)
conda env create -f eval_environment.yml
conda activate eval
```

```bash
# ساخت ایندکس‌ها (بعد از دانلود داده)
chmod +x reset_index.sh
./reset_index.sh
```

```bash
# اجرای سرور — ابتدا WEBSHOP_URL را در WebShopEnv.py تنظیم کنید
conda activate xwebagentbench
python -m web_agent_site.app_multi --log
```

```bash
# ارزیابی
conda activate eval
python Eval/main.py --model gpt-4o --method direct --language zh --test_n 200
```

---

<div dir="rtl">

## روش‌های ارزیابی

</div>

| روش (دقیقاً از کد) | توضیح |
|--------------------|-------|
| `direct` | agent مستقیم با زبان اصلی کار می‌کند |
| `translate_en` | Google Translate صفحات را به انگلیسی و action‌ها را به زبان اصلی برمی‌گرداند |
| `self-translate_en` | خود LLM ترجمه می‌کند (`mode='translate_en'` و `mode='translate'`) |
| `clp_en` | دو مرحله: alignment (`mode='clp'`) → action (`get_clp_2_prompt`) |

---

<div dir="rtl">

## ادغام agent سفارشی

### ساختار generator

generator باید همین signature را داشته باشد (مطابق model.py):

</div>

```python
def my_generator(text, history=[], language='en', mode='direct'):
    # text    : متن مشاهده فعلی
    # history : list از dict های {'role': ..., 'content': ...}
    # language: کد زبان
    # mode    : 'direct' | 'translate' | 'translate_en' | 'clp'
    # return  : (action_string, updated_history)
    ...
    return message, history
```

<div dir="rtl">

### گام ۱ — اضافه کردن به `model.py`

باید داخل `class model()` اضافه شود:

</div>

```python
# در Eval/model.py — داخل class model():
class model():
    # ... توابع موجود ...

    def my_generator(text, history=[], language='en', mode='direct'):
        from WebShop_prompt import get_base_prompt, get_trans_prompt_en, get_trans_prompt, get_clp_base_prompt, get_clp_1_prompt

        if history == [] and mode == 'direct':
            history = [
                {"role": "system", "content": get_base_prompt(language)},
                {"role": "user", "content": text},
            ]
        elif history == [] and mode == 'translate_en':
            history = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": get_trans_prompt_en(text, language)},
            ]
        elif history == [] and mode == 'translate':
            history = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": get_trans_prompt(text, language)},
            ]
        elif history == [] and mode == 'clp':
            history = [
                {"role": "system", "content": get_clp_base_prompt(language)},
                {"role": "user", "content": get_clp_1_prompt(text, language)},
            ]
        else:
            history.append({"role": "user", "content": text})

        # --- اینجا API خود را صدا بزنید ---
        # از OpenAI SDK (با base_url سفارشی):
        client = OpenAI(api_key="YOUR_KEY", base_url="YOUR_BASE_URL")
        response = client.chat.completions.create(
            model="your-model-name",
            messages=history,
        )
        message = response.choices[0].message.content
        # ----------------------------------

        history.append({"role": "system", "content": message})
        return message, history
```

<div dir="rtl">

### گام ۲ — ثبت در `main.py`

</div>

```python
# در Eval/main.py — به بلوک if/elif اضافه کنید:
elif args.model == 'my-model':
    test_model = WebShop_test(model.my_generator, saved_log_path)
```

<div dir="rtl">

### گام ۳ — اجرا

</div>

```bash
python Eval/main.py --model my-model --method direct --language en --test_n 50
```

<div dir="rtl">

### نتایج

در `./saved_log/my-model/direct/direct_en.json` ذخیره می‌شود.
امتیاز نهایی = `total_score * 100` (از ۰ تا ۱۰۰).

</div>

---

<div dir="rtl">

## MTurk در کد

پشتیبانی MTurk از دو بخش تشکیل شده که هر دو در کد هستند:

**۱. تولید کد پاداش** (`web_agent_site/utils.py`):

</div>

```python
def generate_mturk_code(session_id: str) -> str:
    sha = hashlib.sha1(session_id.encode())
    return sha.hexdigest()[:10].upper()
```

<div dir="rtl">

**۲. متن‌های MTurk در رابط کاربری** (`multilingual_text.py`):

</div>

```python
webshop_code  = {'en': 'Your code', 'zh': '您的代码', ...}
webshop_mturk = {'en': 'Paste it in your MTurk interface.', ...}
```

<div dir="rtl">

این کد در `done_page.html` نمایش داده می‌شود. ابزار annotation دستی در `web_agent_site/attributes/annotate.py` موجود است.

</div>

---

<div dir="rtl">

## شخصی‌سازی پیشرفته

### تغییر prompt سیستم

</div>

```python
# در Eval/WebShop_prompt.py، تابع get_base_prompt(language) را ویرایش کنید
def get_base_prompt(language):
    prompt = f'''You are web shopping. ...
    Action: {env_click[language]}[<something>]
    '''
    return prompt
```

<div dir="rtl">

### اضافه کردن زبان جدید

۱. فایل داده بسازید: `data/items_shuffle/items_shuffle_40k_fa.json`

۲. در `web_agent_site/multilingual_text.py` و `Eval/multilingual_text.py` اضافه کنید:

</div>

```python
webshop_search['fa'] = 'جستجو'
webshop_buy['fa'] = 'خرید'
webshop_back_btu['fa'] = 'بازگشت به جستجو'
webshop_prev['fa'] = 'قبلی'
webshop_next['fa'] = 'بعدی'
webshop_description['fa'] = 'توضیحات'
webshop_features['fa'] = 'ویژگی‌ها'
webshop_reviews['fa'] = 'نظرات'
webshop_attributes['fa'] = 'مشخصات'
webshop_score['fa'] = 'امتیاز شما (حداقل 0.0، حداکثر 1.0)'
env_search['fa'] = 'search'    # یا معادل فارسی
env_click['fa'] = 'click'      # یا معادل فارسی
env_attr_click_tip['fa'] = 'شما انتخاب کردید'
```

۳. ایندکس بسازید:

```bash
python search_engine/convert_product_file_format.py --language fa
bash search_engine/run_indexing.sh fa
```

---

<div dir="rtl">

## ۱۰ یافته کلیدی از خواندن کد

### ۱. گلوگاه اصلی: مرحله جستجو
Lucene با BM25 تا ۵۰ نتیجه برمی‌گرداند. اگر agent کلمه اشتباه جستجو کند، محصول هدف اصلاً در لیست نیست. بازیابی بعدی سخت است چون `Back to Search` یک re-search count می‌خورد و محدودیت ۳ بار دارد.

### ۲. r_type ضریب همه‌چیز است
اگر `title_score == 0.0` (هیچ اشتراکی در noun‌ها)، r_type=0.0 و کل reward صفر می‌شود — حتی اگر تمام attributes و options درست باشند.

### ۳. جستجو روی متن lowercase
در `convert_product_file_format.py`، محتوای ایندکس با `.lower()` ذخیره می‌شود. جستجوی Lucene هم case-insensitive عمل می‌کند.

### ۴. قیمت در هر اجرا فرق می‌کند
`generate_product_prices` با `random.uniform` قیمت می‌سازد. پس یک محصول ممکن است در یک اجرا $39.9 و در اجرای بعدی $45.7 باشد. این روی چک قیمت (`price <= price_upper`) تأثیر می‌گذارد.

### ۵. Reviews در کد فعلی غیرفعال است
در `load_products()`، خواندن `reviews.json` comment شده است:
```python
# with open(DEFAULT_REVIEW_PATH) as f:
#     reviews = json.load(f)
```
پس `Rating` همیشه `'N.A.'` و `Reviews` همیشه `[]` است.

### ۶. فرمت پاسخ دوخطی
agent باید `Thought: ...\nAction: ...` برگرداند. `get_action()` با `split('Action: ')[1]` استخراج می‌کند. همچنین اگر `<think>` tag باشد (DeepSeek-Reasoner)، آن را با `split('</think>')[-1]` حذف می‌کند.

### ۷. صفحه‌بندی نتایج
حداکثر ۱۰ صفحه (۵ محصول هر صفحه = ۵۰ محصول). رفتن به صفحه ۱۱ با `assert False` خطا می‌دهد.

### ۸. fuzzy matching نه exact matching
هم attributes و هم options با `fuzz.token_set_ratio > 85` مطابقت می‌یابند. "Water Proof" با "waterproof" match می‌کند.

### ۹. deepseek_v2 پیاده‌سازی نشده
`main.py` به `model.deepseek_v2_generator` ارجاع می‌دهد اما این تابع در `model.py` وجود ندارد — باگ موجود در ریپو.

### ۱۰. ارتباط با WebShop اصلی
محیط بر پایه WebShop (Yao et al. 2022) ساخته شده. ثابت‌های `END_BUTTON = 'Buy Now'`، `NEXT_PAGE = 'Next >'`، `PREV_PAGE = '< Prev'` در `engine.py` از همان پروژه منشأ گرفته‌اند.

</div>

---

<div dir="rtl">

## مرجع سریع فایل‌های کلیدی

</div>

| فایل | نقش |
|------|-----|
| `Eval/main.py` | نقطه ورودی — آرگومان‌های CLI |
| `Eval/model.py` | **اینجا generator خود را در `class model()` اضافه کنید** |
| `Eval/WebShop_prompt.py` | **اینجا prompt را تغییر دهید** |
| `Eval/WebShop_test.py` | حلقه تست — معمولاً نیازی به تغییر ندارد |
| `Eval/WebShopEnv.py` | HTTP client — `WEBSHOP_URL` را تنظیم کنید |
| `Eval/multilingual_text.py` | action‌های چندزبانه |
| `web_agent_site/engine/goal.py` | منطق امتیازدهی |
| `web_agent_site/engine/engine.py` | جستجو + بارگذاری داده |
| `web_agent_site/engine/normalize.py` | نرمال‌سازی رنگ و سایز |
| `web_agent_site/multilingual_text.py` | متن رابط کاربری |
| `web_agent_site/utils.py` | مسیرها + MTurk code |
| `search_engine/convert_product_file_format.py` | تبدیل داده به JSONL |
