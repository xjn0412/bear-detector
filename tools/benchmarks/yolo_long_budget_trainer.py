"""Retain resumable checkpoints before Ultralytics strips optimizer state."""
import json
import shutil
import os
from pathlib import Path
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK


class LongBudgetTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        root = Path(os.environ['YOLO_LONG_BUDGET_ROOT']).resolve()
        overrides = kwargs.get('overrides') or {}
        target = overrides.get('save_dir')
        if not target or not Path(target).resolve().is_relative_to(root):
            raise ValueError('Refusing output directory outside the new long-budget run')
        super().__init__(*args, **kwargs)
        assert self.save_dir.resolve().is_relative_to(root)

    def final_eval(self):
        if RANK in {-1, 0}:
            for src, name in [(self.last, 'last-resume.pt'), (self.best, 'best-resume.pt')]:
                if src.exists():
                    shutil.copyfile(src, self.wdir / name)
            (self.save_dir / 'training-outcome.json').write_text(json.dumps(dict(
                completed_epochs=self.epoch + 1, epoch_limit=self.epochs,
                patience=self.args.patience, best_epoch=self.stopper.best_epoch,
                best_fitness=float(self.stopper.best_fitness),
                ended_before_limit=self.epoch + 1 < self.epochs,
                optimizer=type(self.optimizer).__name__,
                full_resume_checkpoints_preserved=True), indent=2))
        super().final_eval()
