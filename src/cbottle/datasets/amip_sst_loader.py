from cbottle.config import environment as config
from cbottle.storage import get_storage_options

import earth2grid
import torch
import xarray


import datetime


class AmipSSTLoader:
    """AMIP-II SST Forcing dataset

    This is derived from the observed SSTs but is adjusted so that the monthly
    average of linearly interpolated values equals the observed monthly mean. This is
    achieved by solving a linear system enforcing the constraint, with the true obs as the RHS
    This procedure is explained at length here: https://pcmdi.llnl.gov/mips/amip/details/index.html


    # Data Access:

    The data can be downloaded from the Earth System Grid Federation (https://aims2.llnl.gov/)

    The filename is input4MIPs.CMIP6Plus.CMIP.PCMDI.PCMDI-AMIP-1-1-9.ocean.mon.tosbcs.gn

    The unadjusted observed monthly means are input4MIPs.CMIP6Plus.CMIP.PCMDI.PCMDI-AMIP-1-1-9.ocean.mon.tos.gn

    """

    units = "Kelvin"

    def __init__(self, target_grid=None, storage_options=None):
        self.ds = xarray.open_dataset(
            config.AMIP_MID_MONTH_SST,
            engine="h5netcdf",
            storage_options=get_storage_options(config.AMIP_MID_MONTH_SST_PROFILE),
        ).load()

        self.times = self.ds.indexes["time"]

        if target_grid is not None:
            lon_center = self.ds.lon.values
            # need to workaround bug where earth2grid fails to interpolate in circular manner
            # if lon[0] > 0
            # hack: rotate both src and target grids by the same amount so that src_lon[0] == 0
            # See https://github.com/NVlabs/earth2grid/issues/21
            src_lon = lon_center - lon_center[0]
            target_lon = (target_grid.lon - lon_center[0]) % 360

            grid = earth2grid.latlon.LatLonGrid(self.ds.lat.values, src_lon)
            self._regridder = grid.get_bilinear_regridder_to(
                target_grid.lat, lon=target_lon
            )

    async def sel_time(self, times):
        data = self.interp(times)
        return {("tosbcs", -1): self.regrid(data)}

    def interp(self, time: datetime.datetime):
        """Linearly interpolate between the available points"""
        return torch.from_numpy(
            self.ds["tosbcs"].interp(time=time, method="linear").values + 273.15
        )

    def regrid(self, arr):
        return self._regridder(arr)