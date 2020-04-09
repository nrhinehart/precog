
import logging

import interface
import bijection.esp_bijection as esp_bijection

log = logging.getLogger(__file__)

class DynamicRNNBijection(esp_bijection.ESPJointTrajectoryBijectionMixin, interface.ESPJointTrajectoryBijection):
    def _prepare(self, K):
        log.warning("unprepared")

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def step_generate(self, S_history, *args, **kwargs):
        raise NotImplementedError
