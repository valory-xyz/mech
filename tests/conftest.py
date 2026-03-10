# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""Test configuration and fixtures."""

import thinc.util


NUMPY_SEED_MODULUS = 2**32


# Patch thinc's fix_random_seed to clamp seed values to the valid range
# for numpy.random.seed (0 to 2**32 - 1). thinc does not bound the seed
# before passing it to numpy, which causes a ValueError when pytest-randomly
# provides seeds larger than 2**32 - 1 during per-test reseeding.
# See also: https://github.com/explosion/thinc/issues/960 && https://github.com/explosion/thinc/pull/965
_original_fix_random_seed = thinc.util.fix_random_seed


def _fix_random_seed(seed: int = 0) -> None:
    _original_fix_random_seed(seed % NUMPY_SEED_MODULUS)


thinc.util.fix_random_seed = _fix_random_seed
