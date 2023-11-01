# ===========================================================================
# Copyright (C) 2022 Infineon Technologies AG
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================

import numpy as np

class DBF:
    def __init__(self, num_antennas : int, num_beams : int = 27, max_angle_degrees : float = 45, d_by_lambda : float = 0.5):
        """Create a DBF object

        Parameters:
            - num_antennas:         number of (virtual) RX antennas
            - num_beams:            number of beams
            - max_angle_degrees:    maximum angle in degrees, angles will range
                                    from -max_angle_degrees .. +max_angle_degrees
            - d_by_lambda:          separation of RX antennas divided by the wavelength
        """
        angle_vector = np.radians(np.linspace(-max_angle_degrees, max_angle_degrees, num_beams))

        weights = np.zeros((num_antennas, num_beams), dtype=complex)

        for iBeam in range(num_beams):
            angle = angle_vector[iBeam]
            for iAntenna in range(num_antennas):
                weights[iAntenna, iBeam] = np.exp(1j * 2 * np.pi * iAntenna * d_by_lambda * np.sin(angle)) # /sqrt(num_antennas)

        self.weights = weights

    def run(self, range_doppler):
        """Compute virtual beams

        Parameters:
            - range_doppler: Range Doppler spectrum for all RX antennas
              (dimension: num_samples_per_chirp x num_chirps_per_frame x
              num_antennas)
        
        Returns:
            - Range Doppler Beams (dimension: num_samples_per_chirp x
              num_chirps_per_frame x num_beams)
        """

        num_samples, num_chirps, num_antennas = range_doppler.shape
        
        num_antennas_internal, num_beams = self.weights.shape
        
        assert num_antennas == num_antennas_internal
        
        rd_beam_formed = np.zeros((num_samples, num_chirps, num_beams), dtype=complex);

        for iBeam in range(num_beams):
            acc = np.zeros((num_samples, num_chirps), dtype=complex)

            for iAntenna in range(num_antennas):
                acc += range_doppler[:,:,iAntenna] * self.weights[num_antennas-iAntenna-1,iBeam]
            
            rd_beam_formed[:,:,iBeam] = acc

        return rd_beam_formed
