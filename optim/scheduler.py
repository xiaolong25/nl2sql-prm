from transformers import get_cosine_schedule_with_warmup

def build_scheduler(optimizer, total_steps):
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps
    )
