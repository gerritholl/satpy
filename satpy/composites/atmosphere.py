# Copyright (c) 2023- Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Composite classes for derived atmospheric quantities."""

try:
    import metpy
except ImportError:
    metpy = None

from satpy.composites import CompositeBase


def _ensure_metpy():
    """Make sure that metpy is there."""
    if metpy is None:
        raise ImportError("Please install metpy to derive physical quantities from atmospheric profiles.")


class CAPE(CompositeBase):
    """Calculate CAPE from profiles."""

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Perform the CAPE calculation."""
        _ensure_metpy()
        (temperature, air_pressure, water_vapour) = projectables
        raise NotImplementedError()
