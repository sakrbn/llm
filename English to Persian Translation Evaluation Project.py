{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyPeDbcTiWVt4+wmu/3Pc/wm",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/sakrbn/llm/blob/main/English%20to%20Persian%20Translation%20Evaluation%20Project.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "English to Persian Translation Evaluation Project\n",
        "================================================\n",
        "\n",
        "Problem Statement:\n",
        "-----------------\n",
        "This project evaluates English to Persian translation quality using a small dataset of 10 sentences.\n",
        "The goal is to compare the performance of Large Language Models (LLMs) with Google Translate\n",
        "using BLEU score evaluation metrics.\n",
        "\n",
        "Dataset:\n",
        "--------\n",
        "- 10 English sentences with their Persian reference translations\n",
        "- Covers various sentence structures and topics (daily activities, preferences, etc.)\n",
        "\n",
        "Methodology:\n",
        "-----------\n",
        "1. Use Prompt Engineering and In-Context Learning (ICL) without fine-tuning\n",
        "2. Test with preferred models: Aya, Gemma, LLaMA, or similar LLMs\n",
        "3. Compare model translations with Google Translate\n",
        "4. Evaluate using BLEU score metrics\n",
        "5. Provide detailed comparison analysis\n",
        "\n",
        "Models Tested:\n",
        "--------------\n",
        "- Gemma 2B (google/gemma-2b) - LLM with ICL approach\n",
        "- Google Translate API - Baseline comparison\n",
        "\n",
        "Evaluation Metrics:\n",
        "------------------\n",
        "- BLEU Score: Measures translation quality by comparing n-gram overlap\n",
        "- Individual sentence-level BLEU scores\n",
        "- Average BLEU scores across the dataset\n",
        "\n",
        "Expected Outcomes:\n",
        "-----------------\n",
        "- Quantitative comparison between LLM and Google Translate performance\n",
        "- Analysis of translation quality and accuracy\n",
        "- Insights into LLM capabilities for English-Persian translation tasks\n",
        "\n",
        "Results Summary:\n",
        "---------------\n",
        "- Gemma 2B: Struggled with proper Persian translation, produced irrelevant outputs\n",
        "- Google Translate: Achieved significantly higher BLEU scores with accurate translations\n",
        "- Key finding: Specialized translation models outperform general LLMs for this task\n",
        "\n",
        "Author: [Your Name]\n",
        "Date: August 2025\n",
        "Course: [Course Name/Project Context]\n",
        "\"\"\"\n",
        "\n",
        "# Import required libraries\n",
        "import torch\n",
        "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
        "from googletrans import Translator\n",
        "import sacrebleu\n",
        "import re\n",
        "\n",
        "# Your code starts here..."
      ],
      "metadata": {
        "id": "uTuVU4jRgRE8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!huggingface-cli login"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Jj5z-4o88_Ul",
        "outputId": "0e75e5a8-63ad-4195-d114-544f0d828a7c"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[33m⚠️  Warning: 'huggingface-cli login' is deprecated. Use 'hf auth login' instead.\u001b[0m\n",
            "\n",
            "    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|\n",
            "    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|\n",
            "    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|\n",
            "    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|\n",
            "    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|\n",
            "\n",
            "    A token is already saved on your machine. Run `hf auth whoami` to get more information or `hf auth logout` if you want to log out.\n",
            "    Setting a new token will erase the existing one.\n",
            "    To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .\n",
            "Enter your token (input will not be visible): \n",
            "Add token as git credential? (Y/n) Y\n",
            "Token is valid (permission: fineGrained).\n",
            "The token `s` has been saved to /root/.cache/huggingface/stored_tokens\n",
            "\u001b[1m\u001b[31mCannot authenticate through git-credential as no helper is defined on your machine.\n",
            "You might have to re-authenticate when pushing to the Hugging Face Hub.\n",
            "Run the following command in your terminal in case you want to set the 'store' credential helper as default.\n",
            "\n",
            "git config --global credential.helper store\n",
            "\n",
            "Read https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage for more details.\u001b[0m\n",
            "Token has not been saved to git credential helper.\n",
            "Your token has been saved to /root/.cache/huggingface/token\n",
            "Login successful.\n",
            "The current active token is: `s`\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# اصلاح کد محاسبه BLEU Score\n",
        "import torch\n",
        "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
        "from googletrans import Translator\n",
        "import sacrebleu\n",
        "import re\n",
        "\n",
        "# دیتاست (همان قبلی)\n",
        "dataset = {\n",
        "    \"english\": [\n",
        "        \"I woke up early this morning.\",\n",
        "        \"She is reading a very interesting book.\",\n",
        "        \"They went to the park to play football.\",\n",
        "        \"We had dinner at a nice restaurant last night.\",\n",
        "        \"He doesn't like watching horror movies.\",\n",
        "        \"Can you help me with this math problem?\",\n",
        "        \"The weather is getting colder every day.\",\n",
        "        \"I have never been to Paris.\",\n",
        "        \"She always forgets where she puts her keys.\",\n",
        "        \"We are planning a trip to the mountains.\"\n",
        "    ],\n",
        "    \"persian_reference\": [\n",
        "        \"من امروز صبح زود بیدار شدم.\",\n",
        "        \"او دارد یک کتاب بسیار جالب می‌خواند.\",\n",
        "        \"آن‌ها برای بازی فوتبال به پارک رفتند.\",\n",
        "        \"دیشب در یک رستوران خوب شام خوردیم.\",\n",
        "        \"او تماشای فیلم‌های ترسناک را دوست ندارد.\",\n",
        "        \"می‌تونی تو حل این مسئله ریاضی بهم کمک کنی؟\",\n",
        "        \"هوا هر روز سردتر می‌شود.\",\n",
        "        \"من هرگز به پاریس نرفته‌ام.\",\n",
        "        \"او همیشه فراموش می‌کند کلیدهایش را کجا گذاشته.\",\n",
        "        \"ما داریم یک سفر به کوهستان برنامه‌ریزی می‌کنیم.\"\n",
        "    ]\n",
        "}\n",
        "\n",
        "# ترجمه‌های مدل Gemma (از خروجی قبلی)\n",
        "gemma_translations = [\n",
        "    \"من به کتابخانه می‌روم.\",\n",
        "    \"من به فروشگاه می‌رو برای خرید شیر می‌روم.\",\n",
        "    \"من می‌خواهم زبان\",\n",
        "    \"من به سینما با دوستانم می‌روم.\",\n",
        "    \"من به سینما می‌رویم\",\n",
        "    \"من خنچو هستم.\",\n",
        "    \"من دوست دارم فوتبال بازی کنم.\",\n",
        "    \"من خسته ام.\",\n",
        "    \"من دانشجو هستم.\",\n",
        "    \"من دانشجو هستم.\"\n",
        "]\n",
        "\n",
        "# ترجمه‌های Google Translate (از خروجی قبلی)\n",
        "google_translations = [\n",
        "    \"امروز صبح زود از خواب بیدار شدم.\",\n",
        "    \"او در حال خواندن یک کتاب بسیار جالب است.\",\n",
        "    \"آنها برای بازی فوتبال به پارک رفتند.\",\n",
        "    \"دیشب در یک رستوران خوب شام خوردیم.\",\n",
        "    \"او تماشای فیلم های ترسناک را دوست ندارد.\",\n",
        "    \"آیا می توانید در این مشکل ریاضی به من کمک کنید؟\",\n",
        "    \"هوا هر روز سردتر می شود.\",\n",
        "    \"من هرگز به پاریس نرفته ام.\",\n",
        "    \"او همیشه فراموش می کند که کلیدهای خود را قرار می دهد.\",\n",
        "    \"ما در حال برنامه ریزی سفر به کوه هستیم.\"\n",
        "]\n",
        "\n",
        "print(\"📊 محاسبه BLEU Score...\")\n",
        "\n",
        "# تابع محاسبه BLEU Score (اصلاح شده)\n",
        "def calculate_bleu_scores(translations, references):\n",
        "    bleu_scores = []\n",
        "\n",
        "    for i, (trans, ref) in enumerate(zip(translations, references)):\n",
        "        # اطمینان از اینکه ترجمه string است\n",
        "        if not isinstance(trans, str):\n",
        "            trans = str(trans)\n",
        "        if not isinstance(ref, str):\n",
        "            ref = str(ref)\n",
        "\n",
        "        # محاسبه BLEU برای هر جمله\n",
        "        try:\n",
        "            bleu = sacrebleu.sentence_bleu(trans, [ref])\n",
        "            bleu_scores.append(bleu.score)\n",
        "            print(f\"جمله {i+1}: BLEU = {bleu.score:.2f}\")\n",
        "        except Exception as e:\n",
        "            print(f\"خطا در محاسبه BLEU برای جمله {i+1}: {e}\")\n",
        "            bleu_scores.append(0.0)\n",
        "\n",
        "    return bleu_scores\n",
        "\n",
        "# محاسبه BLEU برای هر مدل\n",
        "print(\"\\n🔍 BLEU Score برای مدل Gemma:\")\n",
        "gemma_bleu_scores = calculate_bleu_scores(gemma_translations, dataset[\"persian_reference\"])\n",
        "\n",
        "print(\"\\n🔍 BLEU Score برای Google Translate:\")\n",
        "google_bleu_scores = calculate_bleu_scores(google_translations, dataset[\"persian_reference\"])\n",
        "\n",
        "# محاسبه میانگین\n",
        "avg_gemma_bleu = sum(gemma_bleu_scores) / len(gemma_bleu_scores)\n",
        "avg_google_bleu = sum(google_bleu_scores) / len(google_bleu_scores)\n",
        "\n",
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"📈 نتایج نهایی BLEU Score\")\n",
        "print(\"=\"*60)\n",
        "print(f\"میانگین BLEU Score برای Gemma 2B: {avg_gemma_bleu:.2f}\")\n",
        "print(f\"میانگین BLEU Score برای Google Translate: {avg_google_bleu:.2f}\")\n",
        "print(f\"تفاوت BLEU Score: {abs(avg_google_bleu - avg_gemma_bleu):.2f}\")\n",
        "\n",
        "if avg_google_bleu > avg_gemma_bleu:\n",
        "    print(\"🏆 Google Translate عملکرد بهتری داشته است.\")\n",
        "else:\n",
        "    print(\"🏆 Gemma 2B عملکرد بهتری داشته است.\")\n",
        "\n",
        "print(\"\\n\" + \"=\"*100)\n",
        "print(\"📋 جدول مقایسه تفصیلی\")\n",
        "print(\"=\"*100)\n",
        "\n",
        "for i in range(len(dataset[\"english\"])):\n",
        "    print(f\"\\nجمله {i+1}:\")\n",
        "    print(f\"English: {dataset['english'][i]}\")\n",
        "    print(f\"Reference: {dataset['persian_reference'][i]}\")\n",
        "    print(f\"Gemma: {gemma_translations[i]}\")\n",
        "    print(f\"Google: {google_translations[i]}\")\n",
        "    print(f\"BLEU - Gemma: {gemma_bleu_scores[i]:.2f}\")\n",
        "    print(f\"BLEU - Google: {google_bleu_scores[i]:.2f}\")\n",
        "    print(\"-\" * 100)\n",
        "\n",
        "print(\"\\n🔍 تحلیل نتایج:\")\n",
        "print(\"=\"*60)\n",
        "print(\"❌ مشکلات مدل Gemma 2B:\")\n",
        "print(\"   - خروجی‌های نامربوط (مثل 'من دانشجو هستم' برای جملات مختلف)\")\n",
        "print(\"   - عدم درک صحیح پرامپت ترجمه\")\n",
        "print(\"   - تکرار جملات یکسان\")\n",
        "print(\"   - احتمالاً نیاز به پرامپت بهتر یا مدل بزرگتر\")\n",
        "\n",
        "print(\"\\n✅ مزایای Google Translate:\")\n",
        "print(\"   - ترجمه دقیق و مرتبط\")\n",
        "print(\"   - حفظ معنای اصلی جملات\")\n",
        "print(\"   - کیفیت بالا در زبان فارسی\")\n",
        "\n",
        "print(\"\\n✅ تحلیل کامل شد!\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5_5_z9zzMnhM",
        "outputId": "e6a4408d-a1f2-4b3d-a6f7-65d857adbc97"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "📊 محاسبه BLEU Score...\n",
            "\n",
            "🔍 BLEU Score برای مدل Gemma:\n",
            "جمله 1: BLEU = 8.52\n",
            "جمله 2: BLEU = 4.77\n",
            "جمله 3: BLEU = 0.00\n",
            "جمله 4: BLEU = 5.69\n",
            "جمله 5: BLEU = 0.00\n",
            "جمله 6: BLEU = 0.00\n",
            "جمله 7: BLEU = 6.57\n",
            "جمله 8: BLEU = 11.52\n",
            "جمله 9: BLEU = 4.58\n",
            "جمله 10: BLEU = 4.58\n",
            "\n",
            "🔍 BLEU Score برای Google Translate:\n",
            "جمله 1: BLEU = 34.57\n",
            "جمله 2: BLEU = 29.07\n",
            "جمله 3: BLEU = 84.09\n",
            "جمله 4: BLEU = 100.00\n",
            "جمله 5: BLEU = 51.33\n",
            "جمله 6: BLEU = 4.93\n",
            "جمله 7: BLEU = 43.47\n",
            "جمله 8: BLEU = 43.47\n",
            "جمله 9: BLEU = 13.07\n",
            "جمله 10: BLEU = 9.98\n",
            "\n",
            "============================================================\n",
            "📈 نتایج نهایی BLEU Score\n",
            "============================================================\n",
            "میانگین BLEU Score برای Gemma 2B: 4.62\n",
            "میانگین BLEU Score برای Google Translate: 41.40\n",
            "تفاوت BLEU Score: 36.78\n",
            "🏆 Google Translate عملکرد بهتری داشته است.\n",
            "\n",
            "====================================================================================================\n",
            "📋 جدول مقایسه تفصیلی\n",
            "====================================================================================================\n",
            "\n",
            "جمله 1:\n",
            "English: I woke up early this morning.\n",
            "Reference: من امروز صبح زود بیدار شدم.\n",
            "Gemma: من به کتابخانه می‌روم.\n",
            "Google: امروز صبح زود از خواب بیدار شدم.\n",
            "BLEU - Gemma: 8.52\n",
            "BLEU - Google: 34.57\n",
            "----------------------------------------------------------------------------------------------------\n",
            "\n",
            "جمله 2:\n",
            "English: She is reading a very interesting book.\n",
            "Reference: او دارد یک کتاب بسیار جالب می‌خواند.\n",
            "Gemma: من به فروشگاه می‌رو برای خرید شیر می‌روم.\n",
            "Google: او در حال خواندن یک کتاب بسیار جالب است.\n",
            "BLEU - Gemma: 4.77\n",
            "BLEU - Google: 29.07\n",
            "----------------------------------------------------------------------------------------------------\n",
            "\n",
            "جمله 3:\n",
            "English: They went to the park to play football.\n",
            "Reference: آن‌ها برای بازی فوتبال به پارک رفتند.\n",
            "Gemma: من می‌خواهم زبان\n",
            "Google: آنها برای بازی فوتبال به پارک رفتند.\n",
            "BLEU - Gemma: 0.00\n",
            "BLEU - Google: 84.09\n",
            "----------------------------------------------------------------------------------------------------\n",
            "\n",
            "جمله 4:\n",
            "English: We had dinner at a nice restaurant last night.\n",
            "Reference: دیشب در یک رستوران خوب شام خوردیم.\n",
            "Gemma: من به سینما با دوستانم می‌روم.\n",
            "Google: دیشب در یک رستوران خوب شام خوردیم.\n",
            "BLEU - Gemma: 5.69\n",
            "BLEU - Google: 100.00\n",
            "----------------------------------------------------------------------------------------------------\n",
            "\n",
            "جمله 5:\n",
            "English: He doesn't like watching horror movies.\n",
            "Reference: او تماشای فیلم‌های ترسناک را دوست ندارد.\n",
            "Gemma: من به سینما می‌رویم\n",
            "Google: او تماشای فیلم های ترسناک را دوست ندارد.\n",
            "BLEU - Gemma: 0.00\n",
            "BLEU - Google: 51.33\n",
            "----------------------------------------------------------------------------------------------------\n",
            "\n",
            "جمله 6:\n",
            "English: Can you help me with this math problem?\n",
            "Reference: می‌تونی تو حل این مسئله ریاضی بهم کمک کنی؟\n",
            "Gemma: من خنچو هستم.\n",
            "Google: آیا می توانید در این مشکل ریاضی به من کمک کنید؟\n",
            "BLEU - Gemma: 0.00\n",
            "BLEU - Google: 4.93\n",
            "----------------------------------------------------------------------------------------------------\n",
            "\n",
            "جمله 7:\n",
            "English: The weather is getting colder every day.\n",
            "Reference: هوا هر روز سردتر می‌شود.\n",
            "Gemma: من دوست دارم فوتبال بازی کنم.\n",
            "Google: هوا هر روز سردتر می شود.\n",
            "BLEU - Gemma: 6.57\n",
            "BLEU - Google: 43.47\n",
            "----------------------------------------------------------------------------------------------------\n",
            "\n",
            "جمله 8:\n",
            "English: I have never been to Paris.\n",
            "Reference: من هرگز به پاریس نرفته‌ام.\n",
            "Gemma: من خسته ام.\n",
            "Google: من هرگز به پاریس نرفته ام.\n",
            "BLEU - Gemma: 11.52\n",
            "BLEU - Google: 43.47\n",
            "----------------------------------------------------------------------------------------------------\n",
            "\n",
            "جمله 9:\n",
            "English: She always forgets where she puts her keys.\n",
            "Reference: او همیشه فراموش می‌کند کلیدهایش را کجا گذاشته.\n",
            "Gemma: من دانشجو هستم.\n",
            "Google: او همیشه فراموش می کند که کلیدهای خود را قرار می دهد.\n",
            "BLEU - Gemma: 4.58\n",
            "BLEU - Google: 13.07\n",
            "----------------------------------------------------------------------------------------------------\n",
            "\n",
            "جمله 10:\n",
            "English: We are planning a trip to the mountains.\n",
            "Reference: ما داریم یک سفر به کوهستان برنامه‌ریزی می‌کنیم.\n",
            "Gemma: من دانشجو هستم.\n",
            "Google: ما در حال برنامه ریزی سفر به کوه هستیم.\n",
            "BLEU - Gemma: 4.58\n",
            "BLEU - Google: 9.98\n",
            "----------------------------------------------------------------------------------------------------\n",
            "\n",
            "🔍 تحلیل نتایج:\n",
            "============================================================\n",
            "❌ مشکلات مدل Gemma 2B:\n",
            "   - خروجی‌های نامربوط (مثل 'من دانشجو هستم' برای جملات مختلف)\n",
            "   - عدم درک صحیح پرامپت ترجمه\n",
            "   - تکرار جملات یکسان\n",
            "   - احتمالاً نیاز به پرامپت بهتر یا مدل بزرگتر\n",
            "\n",
            "✅ مزایای Google Translate:\n",
            "   - ترجمه دقیق و مرتبط\n",
            "   - حفظ معنای اصلی جملات\n",
            "   - کیفیت بالا در زبان فارسی\n",
            "\n",
            "✅ تحلیل کامل شد!\n"
          ]
        }
      ]
    }
  ]
}