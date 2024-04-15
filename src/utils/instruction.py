INSTRUCTION_FORMAT = """
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
"""

def format_instruction(sample):
	return INSTRUCTION_FORMAT.format(
        instruction=sample['instruction'],
        input=sample['input'],
        response=sample['output']
    )