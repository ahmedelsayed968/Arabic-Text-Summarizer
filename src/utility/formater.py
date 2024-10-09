from typing import Dict
def alpaca_input_format(entry:Dict[str,str]):
    instruction_text = (
      f"فيما يلي تعليمات تصف مهمة ما. "
      f"اكتب ردًا يكمل الطلب بشكل مناسب."
      f"\n\n### تعليمات:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### النص:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text