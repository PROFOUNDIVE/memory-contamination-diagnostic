import sys
from typing import TYPE_CHECKING

from memcontam.experiment.phase12.filter_challenge import bct_live_impl

if TYPE_CHECKING:
    from memcontam.experiment.phase12.filter_challenge.bct_live_impl import (
        add_calibration_parsers,
        run_calibration_command,
    )

    __all__ = ("add_calibration_parsers", "run_calibration_command")

sys.modules[__name__] = bct_live_impl
