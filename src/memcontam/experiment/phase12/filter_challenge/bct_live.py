import sys

from memcontam.experiment.phase12.filter_challenge import bct_live_impl

sys.modules[__name__] = bct_live_impl
