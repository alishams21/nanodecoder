# Spam Head Fine-tuning Package
# Contains training and evaluation scripts for spam classification fine-tuning

from .spam_head_fine_tuner import main as train_spam_model
from .spam_head_fine_tuner import main as evaluate_spam_model

__all__ = ['train_spam_model', 'evaluate_spam_model']
