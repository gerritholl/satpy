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

import dask
import xarray as xr

from satpy.composites import CompositeBase


def _ensure_metpy():
    """Make sure that metpy is there."""
    if metpy is None:
        raise ImportError("Please install metpy to derive quantities from atmospheric profiles.")


@dask.delayed
def _calculate_cape_cin(air_pressure, temperature, dewpoint):
    """Calculate CAPE and CIN through dask."""
    all_cc = []
    for x in air_pressure.x:
        for y in air_pressure.y:
            p = air_pressure.sel(x=x, y=y)
            t = temperature.sel(x=x, y=y)
            d = dewpoint.sel(x=x, y=y)
            # needs https://github.com/hgrecco/pint/pull/1722 to work!
            pp = dask.delayed(metpy.calc.parcel_profile)(p[::-1], t[-1], d[-1])
            cc = dask.delayed(metpy.calc.cape_cin)(p[::-1], t[::-1], d[::-1], pp)[0]
            all_cc.append(
                    cc.metpy.dequantify.expand_dims(dim={"x": 1, "y": 1}).
                    assign_coords({"x": [x], "y": [x]}))
    return dask.delayed(xr.concat)(all_cc, dim="x").compute()


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
        dewpoint = metpy.calc.dewpoint_from_specific_humidity(
                air_pressure, temperature,
                metpy.calc.mixing_ratio_from_specific_humidity(specific_humidity))

        # calculating CAPE and CIN is neither vectorised nor dask friendly, so
        # use a dask delayed wrapper

        cape = _calculate_cape_cin(air_pressure, temperature, dewpoint)
        return cape.compute()
