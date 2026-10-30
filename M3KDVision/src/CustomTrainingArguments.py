from dataclasses import dataclass, field

@dataclass
class CustomTrainingArguments:
    use_rejected: bool = field(
        default=False, 
        metadata={"help": "Whether to use rejected samples in the loss computation."}
    )
    use_rejected_weight_curriculum: bool = field(
        default=False, 
        metadata={"help": "Whether to use a curriculum for the weight of rejected samples in the loss computation."}
    )
    use_macro_mixing: bool = field(
        default=False,
        metadata={"help": "Whether to use macro mixing in the loss computation."}
    )
    initial_beta: float = field(
        default=0.5, 
        metadata={"help": "Initial beta value for the loss computation."}
    )
    num_micro_samples: int = field(
        default=8, 
        metadata={"help": "Number of micro samples to use in the loss computation."}
    )
