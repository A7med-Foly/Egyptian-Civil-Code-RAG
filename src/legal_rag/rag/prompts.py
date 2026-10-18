"""Prompt templates for Arabic Legal Q&A RAG."""

ARABIC_LEGAL_SYSTEM_PROMPT = """أنت مستشار معلومات قانونية متخصص في القانون المدني المصري.
مهمتك الإجابة عن أسئلة المستخدم بدقة وموضوعية استناداً فقط وحصرياً إلى النصوص القانونية الواردة في السياق المقدم.

القواعد الصارمة التي يجب اتباعها:
1. أجب فقط بناءً على المواد القانونية الواردة في السياق. لا تخترع أو تفترض أي أحكام غير موجودة.
2. اذكر رقم المادة القانونية وسندها بوضوح (مثال: "وفقاً للمادة 147 من القانون المدني المصري...").
3. إذا كانت المادة ملغاة، اذكر بوضوح أن المادة قد ألغيت مع ذكر القرار أو القانون الذي ألغاها.
4. إذا لم يكن السياق كافياً للإجابة عن السؤال، قل بوضوح: "لا يتضمن القانون المدني المصري في النصوص المتوفرة دليلاً كافياً للإجابة عن هذا السؤال."
5. اختم إجابتك دائماً بالتنويه القانوني: "تنويه: هذه معلومات قانونية استرشادية لأغراض تعليمية ولا تُعد بديلاً عن الاستشارة القانونية المهنية."
"""

ARABIC_USER_PROMPT_TEMPLATE = """السياق القانوني المعتمد:
{context}

سؤال المستخدم:
{question}

الإجابة القانونية المستندة للسياق:"""


def format_context_from_chunks(chunks) -> str:
    """Format retrieved legal chunks into clear statutory context with citations."""
    context_blocks = []
    for chunk, score in chunks:
        status_note = " [مادة ملغاة بموجب القانون]" if chunk.is_repealed else ""
        block = (
            f"--- {chunk.citation}{status_note} ---\n"
            f"الكتاب: {chunk.book} | الفصل: {chunk.chapter} | الموضوع: {chunk.topic}\n"
            f"النص:\n{chunk.text}\n"
        )
        context_blocks.append(block)
    return "\n".join(context_blocks)
