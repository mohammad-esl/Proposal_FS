# محیط WebShop

---

## معرفی و منشأ داده

<div dir="rtl">

دیتای اصلی WebShop از scrape کردن amazon.com به دست آمده است. طبق بخش 3.2 مقاله:

- از سرویس ScraperAPI برای استخراج محصولات استفاده شده
- جمعاً **۱٬۱۸۱٬۴۳۶ محصول** (۱.۱۸ میلیون) از ۵ دسته: مد، آرایش، الکترونیک، مبلمان، و غذا
- با استفاده از ۱۱۳ نام زیردسته به‌عنوان query جستجو scrape شده‌اند

</div>

---

## ساختار داده: دو لایه

### لایه ۱ — محصولات (scrape شده، تولید نمی‌شود)

<div dir="rtl">

محصولات واقعی آمازون هستند که یک‌بار استخراج و در فایل‌های JSON ذخیره شده‌اند.

</div>

```python
DEFAULT_ATTR_PATH = join(BASE_DIR, '../data/items_ins_v2_1000.json')
DEFAULT_FILE_PATH = join(BASE_DIR, '../data/items_shuffle_1000.json')
DEFAULT_REVIEW_PATH = join(BASE_DIR, '../data/reviews.json')
```

### لایه ۲ — دستورالعمل‌ها و trajectoryها (تولیدشده توسط انسان)

<div dir="rtl">

- **۱۲٬۰۸۷ دستورالعمل** زبان طبیعی که توسط کارگران Amazon Mechanical Turk (AMT) نوشته شده‌اند
- به هر کارگر یک محصول هدف نشان داده می‌شود (با عنوان، دسته، attributeها، و گزینه‌های خرید) و از او خواسته می‌شود دستوری برای یک agent خرید بنویسد
- میانگین طول: ۱۵.۹ کلمه، واژگان: ۹٬۰۳۶ کلمه
- بیش از **۱٬۶۰۰ trajectory انسانی** (دموی خرید) برای imitation learning استفاده می‌شوند:
  - ۱٬۰۱۲ trajectory آموزشی
  - ۵۴ توسعه
  - ۵۰۰ تست

</div>

---

## سه نسخه از داده

<div dir="rtl">

| نسخه | تعداد محصول | کاربرد |
|------|------------|--------|
| `items_shuffle.json` (با DEBUG_PROD_SIZE) | ۱۰۰ | دیباگ |
| `items_shuffle_1000.json` | ۱٬۰۰۰ | حالت سبک پیش‌فرض کد |
| `items_shuffle.json` کامل | ۱.۱۸ میلیون | نسخه کامل مقاله |

تقسیم‌بندی دستورالعمل‌ها (i.i.d.): train: ۱۰٬۵۸۷ / dev: ۱٬۰۰۰ / test: ۵۰۰

</div>

---

## ساختار فایل‌های داده

### `items_ins` — دستورالعمل‌های synthetic (ماشینی)

```json
"B09Q9561S6": {
  "attributes": ["yoga pants", "butt lifting", "tummy control", ...],
  "instruction": "Find me butt lifting women's pants with tummy control, high waist",
  "instruction_attributes": ["butt lifting", "tummy control", "high waist"]
}
```

<div dir="rtl">

- instruction ماشینی و فرمولیک است: "Find me X, Y with Z"
- فیلد options ندارد
- کلید: asin → object

</div>

### `items_human_ins` — دستورالعمل‌های انسانی

```json
"B015R09D8M": [
  {
    "instruction": "i need a 9.5 rubber soled hiking shoe made of light weight vinyl acetate.",
    "instruction_attributes": ["light weight", "vinyl acetate", "rubber sole"],
    "instruction_options": ["9.5"],
    "options": ["size: 9.5"],
    "worker_id": "A1WS884SI0SLO4",
    "assignment_id": "3S06PH7KS2ESBN3H7VP59DC9HUZ1DI"
  }
]
```

<div dir="rtl">

- instruction طبیعی و محاوره‌ای است
- فیلد `instruction_options` دارد (size/color که باید انتخاب شود)
- یک محصول می‌تواند چند instruction داشته باشد (list)
- `worker_id` و `assignment_id` نشان می‌دهد MTurk کارگران واقعی نوشته‌اند

</div>

### تفاوت کلیدی و کاربرد هر نوع

<div dir="rtl">

| ویژگی | `items_ins` (synthetic) | `items_human_ins` (human) |
|-------|------------------------|--------------------------|
| هدف | پوشش گسترده، مقیاس‌پذیر | ارزیابی واقعی‌تر |
| تولید | TF-IDF خودکار | کارگران MTurk |
| options | ندارد | دارد (size/color) |
| کاربرد | آموزش/pre-training | ارزیابی نهایی benchmark |

**نکته مهم:** دستورالعمل‌های ماشینی بخشی از benchmark مقاله X-WebAgentBench نیستند. مقاله X-WebAgentBench فقط از دستورات انسانی (همان ۵۰۰→۲۰۰ دستور MTurk از ReAct/WebShop) استفاده می‌کند.

</div>

### روش تولید `items_ins` با TF-IDF

<div dir="rtl">

**۱. استخراج attribute با TF-IDF:** روی متن هر محصول (name + small_description) و به‌تفکیک دسته‌بندی، TF-IDF اجرا می‌شود (هم unigram هم bigram، ngram_range از ۱ تا ۲، حداکثر ۱۰۰۰ feature). کلمات/عبارت‌هایی که بالاترین امتیاز TF-IDF را دارند به‌عنوان «ویژگی شاخص» آن محصول انتخاب می‌شوند (k=5 برتر).

**۲. حذف stop words:** هم stop wordهای استاندارد انگلیسی و هم اعداد ۰ تا ۹۹۹ فیلتر می‌شوند تا attributeها معنادار بمانند.

**۳. ساخت instruction از attributeها:** دستور از کنار هم گذاشتن این attributeهای استخراج‌شده ساخته می‌شود.

</div>

---

## عامل‌های پایه (Baseline Agents)

<div dir="rtl">

طبق بخش ۴ مقاله، سه نوع عامل پایه وجود دارد:

</div>

### ۱. Rule Baseline (قاعده‌محور)

<div dir="rtl">

متن دقیق دستور را search می‌کند، اولین محصول نتایج را کلیک و بدون انتخاب گزینه می‌خرد. بدون یادگیری.

</div>

### ۲. IL (Imitation Learning) — عامل اصلی

<div dir="rtl">

ترکیب دو مدل از پیش‌آموزش‌دیده:

- **BART** برای تولید query جستجو (مسئله seq2seq): از دستور u، اکشن `search[...]` تولید می‌کند. با ۱٬۴۲۱ جفت instruction-search از trajectoryهای انسانی fine-tune شده.
- **BERT** (12 لایه، bert-base-uncased) برای انتخاب اکشن کلیک: روی هر اکشن موجود توزیع احتمال می‌سازد. با ۹٬۵۵۸ نمونه از trajectoryها آموزش دیده.
- اختیاری: **ResNet-50** برای پردازش تصویر محصول به بردار ۵۱۲-بُعدی (با فلگ `args.get_image`)

</div>

```python
if args.network == 'rnn':
    self.network = RCDQN(vocab_size, embedding_dim, ...)
elif args.network == 'bert':
    config = BertConfigForWebshop(image=args.get_image,
                                  pretrained_bert=(args.bert_path != 'scratch'))
    self.network = BertModelForWebshop(config)
```

### ۳. IL+RL — بهترین مدل مقاله

<div dir="rtl">

مدل IL با Reinforcement Learning (policy gradient، روش A3C) fine-tune می‌شود. BART فریز می‌شود و top-10 query تولید می‌کند، و BERT یاد می‌گیرد انتخاب کند.

**نتیجه:** Task Score=62.4، SR=28.7%

همه مدل‌ها محلی هستند (PyTorch/HuggingFace) — هیچ API خارجی فراخوانی نمی‌شود.

</div>

---

## روش ارزیابی

<div dir="rtl">

ارزیابی بدون نیاز به انسان و کاملاً خودکار است. دو معیار اصلی:

- **Task Score** = 100 × میانگین reward
- **Success Rate (SR)** = درصد episodeهایی که r = 1 شده

</div>

### فرمول reward

$$r = r_{type} \times \frac{|U_{att} \cap Y_{att}| + |U_{opt} \cap Y_{opt}| + \mathbf{1}[y_{price} \leq u_{price}]}{|U_{att}| + |U_{opt}| + 1}$$

<div dir="rtl">

اجزای فرمول:

- **r_att** (امتیاز attribute): نسبت ویژگی‌های دستور که محصول انتخابی دارد
- **r_option** (امتیاز گزینه): نسبت گزینه‌های درست (رنگ/سایز) که انتخاب شده
- **r_price** (امتیاز قیمت): ۱ اگر قیمت محصول ≤ سقف قیمت دستور
- **r_type**: ضریب TextMatch که وقتی محصول از «نوع» اشتباه باشد reward را پایین می‌آورد

</div>

### مثال عینی محاسبه reward

<div dir="rtl">

هدف برای B015R09D8M:

</div>

```json
{
  "instruction": "i need a 9.5 rubber soled hiking shoe made of light weight vinyl acetate.",
  "instruction_attributes": ["light weight", "vinyl acetate", "rubber sole"],
  "instruction_options": ["9.5"]
}
```

<div dir="rtl">

**نکته مهم:** می‌توان r=1 گرفت حتی اگر محصول نهایی دقیقاً y* نباشد — چون چند محصول می‌توانند هدف «یک پیراهن قرمز» را برآورده کنند.

**گلوگاه اصلی مدل‌ها:** انتخاب گزینه‌های درست (اختلاف ۲۸٪ با انسان) و تولید query جستجوی خوب.

</div>

---

## آنچه عامل می‌بیند (Observation)

<div dir="rtl">

مشاهده‌ی عامل یک متن ساده است که از HTML صفحه استخراج می‌شود. نه HTML خام — فقط متن مرئی با bracket‌های خاص برای عناصر کلیک‌پذیر.

</div>

### صفحه ۱ — شروع (init)

```
WebShop 
Instruction:  
im looking for a earbud headphones for stereo sound quality of style je-04b 
which will be more comfortable for me to use without disturbing others, 
and price lower than 60.00 dollars 
[Search]
```

<div dir="rtl">

عامل فقط دستورالعمل خرید و دکمه‌ی Search می‌بیند.

</div>

### صفحه ۲ — نتایج جستجو (search)

```
Instruction: im looking for ...
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B09743DFJC] 
Jinpei Cute Panda Wireless Earphones, Waterproof, Noise Cancelling...
$39.9 
[B097445J5R] 
Jinpei Cute Pink cat Wireless Earphones...
$39.9 
[B084TSH1YW] 
TWS Headphones Wireless Earbuds...
$59.99 
...
```

<div dir="rtl">

محصولات با ID (مثل B09743DFJC) و قیمت نمایش داده می‌شوند — همه کلیک‌پذیرند.

</div>

### صفحه ۳ — صفحه محصول (item)

```
Instruction: im looking for ...
[Back to Search] 
[< Prev] 
style [je-01b][je-02b][je-03b][[je-04b]][je-05b]
Jinpei Cute Panda Wireless Earphones...
Price: $39.9 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]
```

<div dir="rtl">

**توجه:** آپشن انتخاب‌شده با `[[double bracket]]` نمایش داده می‌شود (مثل `[[je-04b]]`).

</div>

### صفحه ۴ — پایان (end)

```
Your score (min 0.0, max 1.0): 0.6666666666666666
```

<div dir="rtl">

**نکات کلیدی پارسر:** در `WebShopEnv.py:73-93` این parsing انجام می‌شود. عامل در هر مرحله تنها یک صفحه می‌بیند + کل تاریخچه‌ی مکالمه از ابتدا (که مشکل هزینه را ایجاد می‌کند).

</div>

---

> **تصویر ۱:** محیط webshop (در بنچمارک X-WebAgentBench) — صفحه شروع
>
> **تصویر ۲:** نتیجه سرچ در محیط webshop
