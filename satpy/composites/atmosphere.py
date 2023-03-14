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
        raise ImportError("Please install metpy to derive quantities from atmospheric profiles.")


class CAPE(CompositeBase):
    """Calculate CAPE from profiles.

    Needs profiles of temperature, specific humidity, and air pressure.
    """

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Perform the CAPE calculation."""
        _ensure_metpy()
        # cut off the stratosphere, which is irrelevant for CAPE/CIN and leads
        # to problems in calculations involving water vapor mixing ratios
        projectables = [da.sel(vertical_levels=slice(63, None)) for da in projectables]
        (temperature, specific_humidity, air_pressure) = projectables

        # product contains the specific humidity (source:
        # PUG EUM/USC/DOC/22/1281633, § 4.1
        dewp = metpy.calc.dewpoint_from_specific_humidity(
                air_pressure, temperature,
                metpy.calc.mixing_ratio_from_specific_humidity(specific_humidity))

        p = air_pressure.sel(x=50, y=400)
        t = temperature.sel(x=50, y=400)
        d = dewp.sel(x=50, y=400)
        metpy.calc.parcel_profile(p[::-1], t[-1], d[-1])
