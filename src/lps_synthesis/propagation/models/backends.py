import os
import enum

import lps_synthesis.propagation.channel_response as lps_channel_rsp
import lps_synthesis.propagation.models.traceo as traceo
import lps_synthesis.propagation.models.oases as oases

class TypeFactory(enum.Enum):
    """ Enum class to represent available propagation models. """
    TRACEO = 0
    OASES = 1

    def build_model(self, workdir: str = None) -> lps_channel_rsp.SpectralResponse:
        """
        Returns:
            SpectralResponse
        """
        workdir = workdir or os.path.join("./channel", f"{self.name.lower()}")

        if self == TypeFactory.TRACEO:
            return traceo.Traceo(workdir=workdir)

        if self == TypeFactory.OASES:
            return oases.Oases(workdir=workdir)

        else:
            raise NotImplementedError(f"compute_frequency_response not implemented for {self}")

Type = TypeFactory
